#pragma once
// sim_shm.hpp — Simulizer shared-memory output mirror (cross-platform)
//
// Single source of truth for the IPC protocol that mirrors the SSE message
// stream (everything that flows through simulizer::push_message) into a named
// shared-memory ring buffer, so a same-machine consumer (the client/ desktop
// app) can display the same data without going through the HTTP/SSE server.
//
//   Producer (simstd.hpp native build) -> Writer
//   Consumer (client/)                 -> Reader
//
// Each producer/consumer pair agrees on a short "channel" id so multiple
// simulations can run at once on the same machine. The client generates the id
// and passes it to the .sim subprocess via the SIMULIZER_SHM_CHANNEL env var.
//
// Backpressure (single consumer per channel): rather than overwrite data the
// consumer hasn't read, the producer BLOCKS when the ring is full — so nothing
// (not even one-shot series/bar setup) is lost. Two safety guards keep the
// producer from hanging: it only blocks while a consumer is *attached*, and it
// gives up after BACKPRESSURE_TIMEOUT_MS if the consumer stops making progress
// (crashed / frozen UI), falling back to overwrite. The consumer publishes its
// read position + a heartbeat so the producer can see both.
//
// Transport per OS:
//   * Windows : CreateFileMapping/MapViewOfFile + a named auto-reset Event.
//   * POSIX   : shm_open/mmap; the consumer polls (no event object).
// Gated behind !__EMSCRIPTEN__ so the wasm build never sees it.
//
// Layout: a 64-byte header + a CAPACITY-byte byte ring. Records are length-
// prefixed: [uint32 len][len bytes JSON]. `write_seq` is a 64-bit monotonic
// total-bytes-written counter; the physical offset is `write_seq % capacity`.

#if !defined(__EMSCRIPTEN__)

#include <cstdint>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <chrono>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace simulizer {
namespace shm {

// ── protocol constants (platform-independent) ───────────────────────────────
constexpr std::uint32_t MAGIC    = 0x315A4D53u;          // "SMZ1" (LE)
constexpr std::uint32_t VERSION  = 2u;                   // bumped: backpressure header
constexpr std::uint32_t CAPACITY = 4u * 1024u * 1024u;   // 4 MiB ring

// Producer gives up waiting on a stalled/dead consumer after this long and
// resumes (overwriting), so a crashed or frozen client never hangs the sim.
constexpr unsigned BACKPRESSURE_TIMEOUT_MS = 1000;

constexpr const char* DEFAULT_CHANNEL = "default";

// ── OS object names, derived from the channel id ────────────────────────────
#if defined(_WIN32)
inline std::string map_name(const std::string& ch) { return "Local\\Simulizer.Output." + ch; }
inline std::string evt_name(const std::string& ch) { return "Local\\Simulizer.Output." + ch + ".evt"; }
#else
// macOS limits shm names to ~31 chars incl. the leading '/', so keep it short.
inline std::string shm_name(const std::string& ch) { return "/smlz." + ch; }
#endif

#pragma pack(push, 8)
struct Header {
    std::uint32_t magic;            // = MAGIC
    std::uint32_t version;          // = VERSION
    std::uint32_t capacity;         // = CAPACITY (ring size in bytes)
    std::uint32_t reserved;
    std::uint64_t write_seq;        // total bytes written (producer)
    std::uint64_t drop_count;       // records dropped (too large, or timed-out backpressure)
    std::uint64_t reader_seq;       // bytes consumed (consumer) — for backpressure
    std::uint64_t reader_hb;        // consumer heartbeat (liveness)
    std::uint32_t reader_attached;  // 1 while a consumer is attached
    std::uint8_t  pad[12];          // -> 64-byte header
};
#pragma pack(pop)
static_assert(sizeof(Header) == 64, "Header layout drift");

constexpr std::size_t TOTAL_SIZE = sizeof(Header) + CAPACITY;

// ── ring copy helpers (handle wrap across the ring boundary) ────────────────
inline void ring_write(std::uint8_t* data, std::uint32_t cap,
                       std::uint64_t seq, const void* src, std::uint32_t n) {
    const std::uint32_t off   = static_cast<std::uint32_t>(seq % cap);
    const std::uint32_t first = (n < cap - off) ? n : (cap - off);
    std::memcpy(data + off, src, first);
    if (n > first)
        std::memcpy(data, static_cast<const std::uint8_t*>(src) + first, n - first);
}

inline void ring_read(const std::uint8_t* data, std::uint32_t cap,
                      std::uint64_t seq, void* dst, std::uint32_t n) {
    const std::uint32_t off   = static_cast<std::uint32_t>(seq % cap);
    const std::uint32_t first = (n < cap - off) ? n : (cap - off);
    std::memcpy(dst, data + off, first);
    if (n > first)
        std::memcpy(static_cast<std::uint8_t*>(dst) + first, data, n - first);
}

inline void sleep_ms(unsigned ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ── Writer — producer side (used by simstd.hpp) ─────────────────────────────
class Writer {
#if defined(_WIN32)
    HANDLE        map_  = nullptr;
    HANDLE        evt_  = nullptr;
#else
    int           fd_   = -1;
#endif
    Header*       h_    = nullptr;
    std::uint8_t* data_ = nullptr;
    std::mutex    mtx_;   // intra-process: serializes the main + sampler threads

    void init_header_if_needed(bool fresh) {
        if (!fresh && h_->magic == MAGIC) return;
        h_->magic      = MAGIC;
        h_->version    = VERSION;
        h_->capacity   = CAPACITY;
        h_->reserved   = 0;
        h_->drop_count = 0;
        h_->reader_seq = 0;
        h_->reader_hb  = 0;
        __atomic_store_n(&h_->write_seq, static_cast<std::uint64_t>(0), __ATOMIC_RELEASE);
    }

    // Block until the ring has room for `total` bytes past `seq` without
    // overwriting unread data — but only while a consumer is attached and
    // making progress. Returns once there's room or we've given up.
    void apply_backpressure(std::uint64_t seq, std::uint64_t total, std::uint32_t cap) {
        if (!__atomic_load_n(&h_->reader_attached, __ATOMIC_ACQUIRE)) return;
        std::uint64_t last_hb = __atomic_load_n(&h_->reader_hb, __ATOMIC_ACQUIRE);
        unsigned stalled_ms = 0;
        for (;;) {
            const std::uint64_t rseq = __atomic_load_n(&h_->reader_seq, __ATOMIC_ACQUIRE);
            if (seq + total - rseq <= cap) return;                     // room available
            if (!__atomic_load_n(&h_->reader_attached, __ATOMIC_ACQUIRE)) return;  // consumer left
            sleep_ms(1);
            const std::uint64_t hb = __atomic_load_n(&h_->reader_hb, __ATOMIC_ACQUIRE);
            if (hb != last_hb) { last_hb = hb; stalled_ms = 0; }       // consumer alive
            else if (++stalled_ms >= BACKPRESSURE_TIMEOUT_MS) {        // stuck/dead: give up
                ++h_->drop_count;
                return;
            }
        }
    }

public:
    Writer() = default;
    Writer(const Writer&)            = delete;
    Writer& operator=(const Writer&) = delete;

    bool init(const std::string& channel = DEFAULT_CHANNEL) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (h_) return true;

#if defined(_WIN32)
        const std::string mapn = map_name(channel);
        map_ = CreateFileMappingA(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
            static_cast<DWORD>(TOTAL_SIZE >> 32),
            static_cast<DWORD>(TOTAL_SIZE & 0xFFFFFFFFu),
            mapn.c_str());
        if (!map_) return false;
        const bool fresh = (GetLastError() != ERROR_ALREADY_EXISTS);

        void* base = MapViewOfFile(map_, FILE_MAP_ALL_ACCESS, 0, 0, TOTAL_SIZE);
        if (!base) { CloseHandle(map_); map_ = nullptr; return false; }

        h_    = reinterpret_cast<Header*>(base);
        data_ = reinterpret_cast<std::uint8_t*>(base) + sizeof(Header);
        init_header_if_needed(fresh);
        // No consumer at producer start (channel ids are unique per run).
        __atomic_store_n(&h_->reader_attached, 0u, __ATOMIC_RELEASE);

        evt_ = CreateEventA(nullptr, FALSE /*auto-reset*/, FALSE, evt_name(channel).c_str());
        return true;
#else
        const std::string shmn = shm_name(channel);
        fd_ = ::shm_open(shmn.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0) return false;
        if (::ftruncate(fd_, static_cast<off_t>(TOTAL_SIZE)) != 0) {
            ::close(fd_); fd_ = -1; return false;
        }
        void* base = ::mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (base == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }

        h_    = reinterpret_cast<Header*>(base);
        data_ = reinterpret_cast<std::uint8_t*>(base) + sizeof(Header);
        init_header_if_needed(/*fresh=*/false);  // magic check decides
        __atomic_store_n(&h_->reader_attached, 0u, __ATOMIC_RELEASE);
        return true;
#endif
    }

    // Append one record (blocks under backpressure). No-op until init().
    void publish(const std::string& msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!h_) return;

        const std::uint32_t len   = static_cast<std::uint32_t>(msg.size());
        const std::uint64_t total = static_cast<std::uint64_t>(4) + len;
        const std::uint32_t cap   = h_->capacity;
        if (total > cap) { ++h_->drop_count; return; }   // record larger than ring

        const std::uint64_t seq = h_->write_seq;
        apply_backpressure(seq, total, cap);             // wait for the consumer if needed

        ring_write(data_, cap, seq, &len, 4);
        ring_write(data_, cap, seq + 4, msg.data(), len);
        __atomic_store_n(&h_->write_seq, seq + total, __ATOMIC_RELEASE);

#if defined(_WIN32)
        if (evt_) SetEvent(evt_);
#endif
    }

    ~Writer() {
#if defined(_WIN32)
        if (h_)   UnmapViewOfFile(h_);
        if (map_) CloseHandle(map_);
        if (evt_) CloseHandle(evt_);
#else
        if (h_)        ::munmap(h_, TOTAL_SIZE);
        if (fd_ >= 0)  ::close(fd_);
#endif
    }
};

// ── Reader — consumer side (used by client/) ────────────────────────────────
// Maps the segment read-write: it only reads ring data, but writes its progress
// (reader_seq / reader_hb / reader_attached) into the header so the producer's
// backpressure can see it. Single consumer per channel.
class Reader {
#if defined(_WIN32)
    HANDLE              map_  = nullptr;
    HANDLE              evt_  = nullptr;
#else
    int                 fd_   = -1;
#endif
    Header*             h_    = nullptr;   // RW (header progress fields)
    const std::uint8_t* data_ = nullptr;   // ring data is read-only to us
    std::uint64_t       read_seq_ = 0;
    std::uint64_t       drops_    = 0;

    void publish_progress() {
        if (h_) __atomic_store_n(&h_->reader_seq, read_seq_, __ATOMIC_RELEASE);
    }

public:
    Reader() = default;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    bool open(const std::string& channel = DEFAULT_CHANNEL) {
        if (h_) return true;

#if defined(_WIN32)
        map_ = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, map_name(channel).c_str());
        if (!map_) return false;
        void* base = MapViewOfFile(map_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, TOTAL_SIZE);
        if (!base) { CloseHandle(map_); map_ = nullptr; return false; }
        Header* h = reinterpret_cast<Header*>(base);
        if (h->magic != MAGIC || h->version != VERSION) {
            UnmapViewOfFile(base); CloseHandle(map_); map_ = nullptr; return false;
        }
        h_    = h;
        data_ = reinterpret_cast<const std::uint8_t*>(base) + sizeof(Header);
        evt_  = OpenEventA(SYNCHRONIZE, FALSE, evt_name(channel).c_str());
#else
        fd_ = ::shm_open(shm_name(channel).c_str(), O_RDWR, 0);
        if (fd_ < 0) return false;
        struct stat st;
        if (::fstat(fd_, &st) != 0 ||
            static_cast<std::size_t>(st.st_size) < TOTAL_SIZE) {
            ::close(fd_); fd_ = -1; return false;   // producer still sizing it
        }
        void* base = ::mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (base == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }
        Header* h = reinterpret_cast<Header*>(base);
        if (h->magic != MAGIC || h->version != VERSION) {
            ::munmap(base, TOTAL_SIZE); ::close(fd_); fd_ = -1; return false;
        }
        h_    = h;
        data_ = reinterpret_cast<const std::uint8_t*>(base) + sizeof(Header);
#endif
        // Read from the start so one-shot setup (series/bar) is delivered; with
        // backpressure the producer won't lap us once we're attached.
        read_seq_ = 0;
        __atomic_store_n(&h_->reader_seq, read_seq_, __ATOMIC_RELEASE);
        __atomic_store_n(&h_->reader_hb, static_cast<std::uint64_t>(0), __ATOMIC_RELEASE);
        __atomic_store_n(&h_->reader_attached, 1u, __ATOMIC_RELEASE);
        return true;
    }

    bool next(std::string& out) {
        if (!h_) return false;
        const std::uint32_t cap = h_->capacity;

        for (;;) {
            const std::uint64_t w = __atomic_load_n(&h_->write_seq, __ATOMIC_ACQUIRE);
            if (read_seq_ == w) return false;          // caught up
            if (w - read_seq_ > cap) {                 // overrun (backpressure timed out)
                ++drops_;
                read_seq_ = w;
                publish_progress();
                continue;
            }

            const std::uint64_t start = read_seq_;
            std::uint32_t len = 0;
            ring_read(data_, cap, start, &len, 4);
            const std::uint64_t total = static_cast<std::uint64_t>(4) + len;

            if (len > cap || start + total > w) {
                const std::uint64_t w2 = __atomic_load_n(&h_->write_seq, __ATOMIC_ACQUIRE);
                ++drops_; read_seq_ = w2; publish_progress(); continue;
            }

            out.resize(len);
            if (len) ring_read(data_, cap, start + 4, &out[0], len);

            const std::uint64_t w2 = __atomic_load_n(&h_->write_seq, __ATOMIC_ACQUIRE);
            if (w2 - start > cap) { ++drops_; read_seq_ = w2; publish_progress(); continue; }

            read_seq_ += total;
            publish_progress();   // let the producer reclaim space promptly
            return true;
        }
    }

    // Block up to `ms` for new data, and emit a heartbeat so the producer's
    // backpressure knows we're alive. Always re-check next() after returning.
    void wait(unsigned ms) {
        if (h_) {
            __atomic_fetch_add(&h_->reader_hb, static_cast<std::uint64_t>(1), __ATOMIC_RELEASE);
            publish_progress();
        }
#if defined(_WIN32)
        if (evt_) { WaitForSingleObject(evt_, ms); return; }
#endif
        sleep_ms(ms);
    }

    std::uint64_t drops() const { return drops_; }

    ~Reader() {
        if (h_) __atomic_store_n(&h_->reader_attached, 0u, __ATOMIC_RELEASE);  // unblock producer
#if defined(_WIN32)
        if (h_)   UnmapViewOfFile(h_);
        if (map_) CloseHandle(map_);
        if (evt_) CloseHandle(evt_);
#else
        if (h_)       ::munmap(h_, TOTAL_SIZE);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
};

}  // namespace shm
}  // namespace simulizer

#endif  // !__EMSCRIPTEN__
