#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/io_uring.h>
#include <linux/fs.h>
#include <random>
#include <getopt.h>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#define QUEUE_DEPTH_DEFAULT 64
#define BLOCK_SZ_DEFAULT 4096

#define KILO 1024
#define KIBI 1024

/* Memory barriers */
#define read_barrier() __asm__ __volatile__("" ::: "memory")
#define write_barrier() __asm__ __volatile__("" ::: "memory")

struct app_io_sq_ring
{
    unsigned *head;
    unsigned *tail;
    unsigned *ring_mask;
    unsigned *ring_entries;
    unsigned *flags;
    unsigned *array;
};

struct app_io_cq_ring
{
    unsigned *head;
    unsigned *tail;
    unsigned *ring_mask;
    unsigned *ring_entries;
    struct io_uring_cqe *cqes;
};

struct submitter
{
    int ring_fd;
    void *sq_ptr;
    void *cq_ptr;
    size_t sring_sz;
    size_t cring_sz;
    size_t sqes_sz;
    struct app_io_sq_ring sq_ring;
    struct io_uring_sqe *sqes;
    struct app_io_cq_ring cq_ring;
};

struct io_data
{
    void *buf;
    off_t offset;
    size_t length;
};

std::string byte_conversion(unsigned long long bytes, const std::string &unit)
{
    const std::string binary[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    const std::string metric[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    const auto &units = (unit == "binary") ? binary : metric;
    int i = 0;
    unsigned long long base = (unit == "binary") ? KIBI : KILO;

    while (bytes >= base && i < 5)
    {
        bytes /= base;
        i++;
    }
    return std::to_string(static_cast<int>(round(bytes))) + " " + units[i];
}

/**
 * @brief Performs the io_uring_setup system call.
 *
 * @param entries Number of submission queue entries.
 * @param p Pointer to io_uring_params structure.
 * @return File descriptor on success, -1 on failure.
 */
int io_uring_setup(unsigned entries, struct io_uring_params *p)
{
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

/**
 * @brief Performs the io_uring_enter system call.
 *
 * @param ring_fd File descriptor of the io_uring instance.
 * @param to_submit Number of submissions to submit.
 * @param min_complete Minimum number of completions.
 * @param flags Flags for the system call.
 * @return Number of events submitted on success, -1 on failure.
 */
int io_uring_enter(int ring_fd, unsigned int to_submit,
                   unsigned int min_complete, unsigned int flags)
{
    return (int)syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete,
                        flags, NULL, 0);
}

/**
 * @brief Sets up the io_uring instance and maps the rings.
 *
 * @param s Pointer to the submitter structure.
 * @param queue_depth Depth of the submission queue.
 * @return 0 on success, 1 on failure.
 */
int app_setup_uring(struct submitter *s, unsigned queue_depth)
{
    struct app_io_sq_ring *sring = &s->sq_ring;
    struct app_io_cq_ring *cring = &s->cq_ring;
    struct io_uring_params p;
    void *sq_ptr, *cq_ptr;

    memset(&p, 0, sizeof(p));
    s->ring_fd = io_uring_setup(queue_depth, &p);
    if (s->ring_fd < 0)
    {
        perror("io_uring_setup");
        return 1;
    }

    s->sring_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    s->cring_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);

    if (p.features & IORING_FEAT_SINGLE_MMAP)
    {
        if (s->cring_sz > s->sring_sz)
        {
            s->sring_sz = s->cring_sz;
        }
        s->cring_sz = s->sring_sz;
    }

    sq_ptr = mmap(0, s->sring_sz, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_POPULATE,
                  s->ring_fd, IORING_OFF_SQ_RING);
    if (sq_ptr == MAP_FAILED)
    {
        perror("mmap");
        return 1;
    }
    s->sq_ptr = sq_ptr;

    if (p.features & IORING_FEAT_SINGLE_MMAP)
    {
        cq_ptr = sq_ptr;
    }
    else
    {
        cq_ptr = mmap(0, s->cring_sz, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE,
                      s->ring_fd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED)
        {
            perror("mmap");
            munmap(sq_ptr, s->sring_sz);
            return 1;
        }
        s->cq_ptr = cq_ptr;
    }

    /* Correct pointer calculations */
    sring->head = (unsigned *)((char *)sq_ptr + p.sq_off.head);
    sring->tail = (unsigned *)((char *)sq_ptr + p.sq_off.tail);
    sring->ring_mask = (unsigned *)((char *)sq_ptr + p.sq_off.ring_mask);
    sring->ring_entries = (unsigned *)((char *)sq_ptr + p.sq_off.ring_entries);
    sring->flags = (unsigned *)((char *)sq_ptr + p.sq_off.flags);
    sring->array = (unsigned *)((char *)sq_ptr + p.sq_off.array);

    s->sqes_sz = p.sq_entries * sizeof(struct io_uring_sqe);
    s->sqes = (struct io_uring_sqe *)mmap(0, s->sqes_sz,
                                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                                          s->ring_fd, IORING_OFF_SQES);

    if (s->sqes == MAP_FAILED)
    {
        perror("mmap");
        munmap(s->sq_ptr, s->sring_sz);
        if (!(p.features & IORING_FEAT_SINGLE_MMAP))
        {
            munmap(s->cq_ptr, s->cring_sz);
        }
        return 1;
    }

    cring->head = (unsigned *)((char *)cq_ptr + p.cq_off.head);
    cring->tail = (unsigned *)((char *)cq_ptr + p.cq_off.tail);
    cring->ring_mask = (unsigned *)((char *)cq_ptr + p.cq_off.ring_mask);
    cring->ring_entries = (unsigned *)((char *)cq_ptr + p.cq_off.ring_entries);
    cring->cqes = (struct io_uring_cqe *)((char *)cq_ptr + p.cq_off.cqes);

    return 0;
}

/**
 * @brief Submits an I/O operation using io_uring.
 *
 * @param s Pointer to the submitter structure.
 * @param fd File descriptor of the device.
 * @param block_size Size of each I/O operation.
 * @param offset Offset for the I/O operation.
 * @param is_read True for read, false for write.
 * @param io Pointer to the io_data structure.
 */
void submit_io(struct submitter *s, int fd, size_t block_size, off_t offset, bool is_read, struct io_data *io)
{
    struct app_io_sq_ring *sring = &s->sq_ring;
    unsigned tail, index;

    tail = *sring->tail;
    index = tail & *sring->ring_mask;

    struct io_uring_sqe *sqe = &s->sqes[index];
    memset(sqe, 0, sizeof(*sqe));

    io->length = block_size;
    io->offset = offset;
    if (posix_memalign(&io->buf, 4096, block_size))
    { // Align to 4096 bytes
        perror("posix_memalign");
        exit(1);
    }

    sqe->fd = fd;
    sqe->addr = (unsigned long)io->buf;
    sqe->len = block_size;
    sqe->off = offset;
    sqe->user_data = (unsigned long long)io;

    if (is_read)
    {
        sqe->opcode = IORING_OP_READ;
    }
    else
    {
        sqe->opcode = IORING_OP_WRITE;
        memset(io->buf, 0xAA, block_size); // Fill buffer with dummy data
    }

    sring->array[index] = index;
    tail++;
    *sring->tail = tail;
    write_barrier();
}

/**
 * @brief Reaps completed I/O operations from the completion queue.
 *
 * @param s Pointer to the submitter structure.
 * @param completed_ios Atomic counter for completed I/Os.
 * @param total_bytes Atomic counter for total bytes transferred.
 */
void reap_cqes(struct submitter *s, std::atomic<unsigned> &completed_ios, std::atomic<unsigned long long> &total_bytes)
{
    struct app_io_cq_ring *cring = &s->cq_ring;
    unsigned head;

    head = *cring->head;

    while (head != *cring->tail)
    {
        read_barrier();
        struct io_uring_cqe *cqe = &cring->cqes[head & *cring->ring_mask];
        struct io_data *io = (struct io_data *)cqe->user_data;

        if (cqe->res < 0)
        {
            std::cerr << "I/O error: " << strerror(-cqe->res) << std::endl;
        }
        else if ((size_t)cqe->res != io->length)
        {
            std::cerr << "Partial I/O: " << cqe->res << " bytes" << std::endl;
        }

        free(io->buf);
        delete io;

        head++;
        completed_ios++;
        total_bytes += io->length;
    }

    *cring->head = head;
    write_barrier();
}

/**
 * @brief Parses command-line arguments.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param device Reference to device string.
 * @param queue_depth Reference to queue_depth.
 * @param duration_sec Reference to duration in seconds.
 * @param block_size Reference to block_size.
 * @param is_read Reference to is_read flag.
 * @param is_random Reference to is_random flag.
 * @return 0 on success, 1 on failure.
 */
int parse_arguments(int argc, char *argv[],
                    std::string &device,
                    unsigned &queue_depth,
                    unsigned &duration_sec,
                    size_t &block_size,
                    bool &is_read,
                    bool &is_random)
{
    int option_index = 0;
    int c;
    static struct option long_options[] = {
        {"device", required_argument, 0, 0},
        {"queue_depth", required_argument, 0, 0},
        {"duration", required_argument, 0, 0},
        {"page_size", required_argument, 0, 0},
        {"operation", required_argument, 0, 0},
        {"method", required_argument, 0, 0},
        {0, 0, 0, 0}};

    // Set default values
    queue_depth = QUEUE_DEPTH_DEFAULT;
    block_size = BLOCK_SZ_DEFAULT;

    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1)
    {
        if (c == 0)
        {
            std::string opt_name = long_options[option_index].name;
            std::string opt_value = optarg;

            if (opt_name == "device")
            {
                device = opt_value;
            }
            else if (opt_name == "queue_depth")
            {
                queue_depth = std::stoul(opt_value);
            }
            else if (opt_name == "duration")
            {
                duration_sec = std::stoul(opt_value);
            }
            else if (opt_name == "page_size")
            {
                block_size = std::stoul(opt_value);
            }
            else if (opt_name == "operation")
            {
                if (opt_value == "read")
                {
                    is_read = true;
                }
                else if (opt_value == "write")
                {
                    is_read = false;
                }
                else
                {
                    std::cerr << "Invalid operation. Use 'read' or 'write'." << std::endl;
                    return 1;
                }
            }
            else if (opt_name == "method")
            {
                if (opt_value == "seq")
                {
                    is_random = false;
                }
                else if (opt_value == "rand")
                {
                    is_random = true;
                }
                else
                {
                    std::cerr << "Invalid method. Use 'seq' or 'rand'." << std::endl;
                    return 1;
                }
            }
        }
        else
        {
            std::cerr << "Invalid option." << std::endl;
            return 1;
        }
    }

    // Check mandatory arguments
    if (device.empty() || duration_sec == 0)
    {
        std::cerr << "Usage: " << argv[0] << " --device=<device> --queue_depth=<queue_depth> --duration=<duration_sec> --page_size=<block_size> --operation=read/write --method=seq/rand" << std::endl;
        return 1;
    }

    return 0;
}

/**
 * @brief Runs the I/O benchmark based on provided parameters.
 *
 * @param device Device path.
 * @param queue_depth Depth of the submission queue.
 * @param duration_sec Duration of the test in seconds.
 * @param block_size Size of each I/O operation.
 * @param is_read True for read, false for write.
 * @param is_random True for random access, false for sequential.
 * @return 0 on success, 1 on failure.
 */
int run_benchmark(const std::string &device,
                  unsigned queue_depth,
                  unsigned duration_sec,
                  size_t block_size,
                  bool is_read,
                  bool is_random)
{
    int fd = open(device.c_str(), O_RDWR | O_DIRECT | O_SYNC);
    if (fd < 0)
    {
        perror("open");
        return 1;
    }

    struct submitter *s = new submitter();
    memset(s, 0, sizeof(*s));

    if (app_setup_uring(s, queue_depth))
    {
        std::cerr << "Unable to setup io_uring!" << std::endl;
        close(fd);
        delete s;
        return 1;
    }

    off_t device_size = lseek(fd, 0, SEEK_END);
    if (device_size <= 0)
    {
        perror("lseek");
        close(fd);
        delete s;
        return 1;
    }
    lseek(fd, 0, SEEK_SET);

    // Ensure device_size is a multiple of block_size
    device_size = (device_size / block_size) * block_size;

    // Random number generator setup
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<off_t> dist(0, device_size - block_size);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Variables for submission tracking
    unsigned submitted_ios = 0;
    unsigned to_submit = 0;
    off_t offset = 0;

    std::ostringstream stats_buffer;
    std::atomic<unsigned int> completed_ios = 0;
    std::atomic<unsigned long long> total_bytes = 0;

    while (true)
    {
        // Check if duration has passed
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed_sec >= duration_sec)
        {
            break;
        }

        // Submit as many I/Os as possible up to queue_depth
        while ((submitted_ios - completed_ios) < queue_depth)
        {
            struct io_data *io = new io_data();

            // Generate offset
            if (is_random)
            {
                offset = dist(rng);
                // Align offset to block_size
                offset = (offset / block_size) * block_size;
            }
            else
            {
                // Sequential
                offset += block_size;
                if (offset >= device_size)
                {
                    offset = 0;
                }
            }

            submit_io(s, fd, block_size, offset, is_read, io);
            submitted_ios++;
            to_submit++;
        }

        // Submit to the kernel
        int ret = io_uring_enter(s->ring_fd, to_submit, 0, 0);
        if (ret < 0)
        {
            perror("io_uring_enter");
            close(fd);
            delete s;
            return 1;
        }
        to_submit = 0; // Reset after submission

        // Process completions
        reap_cqes(s, completed_ios, total_bytes);

        // Update stats
        if (submitted_ios % 5000 == 0)
        {
            stats_buffer.str("");
            stats_buffer << "\rI/Os: " << completed_ios << ", Bytes: " << total_bytes << ", Time: " << elapsed_sec << " sec";
            std::cout << stats_buffer.str() << std::flush;
        }
    }
    std::cout << "\n";

    // Final processing of any remaining completions
    while (completed_ios < submitted_ios)
    {
        reap_cqes(s, completed_ios, total_bytes);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    double bandwidth = (double)total_bytes / (1024.0 * 1024.0) / total_time_sec; // MB/s
    double iops = (double)completed_ios / total_time_sec;

    std::cout << "Test completed." << std::endl;
    std::cout << "Total I/Os: " << completed_ios << std::endl;
    std::cout << "Total bytes: " << total_bytes << " bytes" << " (" << byte_conversion(total_bytes, "binary") << ")" << std::endl;
    std::cout << "Total time: " << total_time_sec << " seconds" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " MB/s" << std::endl;
    std::cout << "IOPS: " << iops << " ops/sec" << std::endl;

    // Clean up
    close(fd);
    munmap(s->sq_ptr, s->sring_sz);
    if (s->cq_ptr && s->cq_ptr != s->sq_ptr)
    {
        munmap(s->cq_ptr, s->cring_sz);
    }
    munmap(s->sqes, s->sqes_sz);
    close(s->ring_fd);
    delete s;

    return 0;
}

int main(int argc, char *argv[])
{
    std::string device;
    unsigned queue_depth = QUEUE_DEPTH_DEFAULT;
    unsigned duration_sec = 0;
    size_t block_size = BLOCK_SZ_DEFAULT;
    bool is_read = true;
    bool is_random = false;

    if (parse_arguments(argc, argv, device, queue_depth, duration_sec, block_size, is_read, is_random))
    {
        return 1;
    }

    if (run_benchmark(device, queue_depth, duration_sec, block_size, is_read, is_random))
    {
        return 1;
    }

    return 0;
}
