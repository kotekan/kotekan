#include "../include/pirate/file_utils.hpp"
#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <fts.h>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

namespace pirate {
#if 0
}   // compiler pacifier
#endif


// -------------------------------------------------------------------------------------------------


// Helper: Takes an existing error, formats a new message, and re-throws
// Usage: rethrow_verbose(e, "my_func", path1, [path2])
template <typename... Args>
[[noreturn]] static void rethrow_verbose(const fs::filesystem_error& e, const char* func_name, Args&&... args) 
{
    std::ostringstream msg;
    msg << func_name << "(";
    
    // Fold expression to join args with commas (C++17)
    const char* sep = "";
    ((msg << sep << "'" << args.string() << "'", sep = ", "), ...);
    
    msg << ")";
    
    // Throw new error with: "func('arg'): Original Error", preserving path/code
    throw fs::filesystem_error(msg.str(), e.path1(), e.path2(), e.code());
}


// Returns true if a new directory was created, false otherwise.
bool create_directories(const fs::path& p) 
{
    try {
        return fs::create_directories(p);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "create_directories", p);
    }
}

// Returns true if the file was copied, false otherwise.
bool copy_file(const fs::path &from, const fs::path &to, fs::copy_options options) 
{    
    try {
        return fs::copy_file(from, to, options);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "copy_file", from, to);
    }
}

// Returns true if file was deleted, false if it didn't exist.
bool remove_file(const fs::path& p) 
{
    try {
        return fs::remove(p);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "remove", p);
    }
}

// Note: On POSIX, this is atomic. If 'to' already exists, it is replaced.
void rename_file(const fs::path& from, const fs::path& to) {
    try {
        fs::rename(from, to);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "rename", from, to);
    }
}

void create_hard_link(const fs::path& target, const fs::path& link) 
{
    try {
        fs::create_hard_link(target, link);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "create_hard_link", target, link);
    }
}

bool file_exists(const fs::path &p) 
{
    try {
        return fs::exists(p);
    } catch (const fs::filesystem_error& e) {
        rethrow_verbose(e, "exists", p);
    }
}

bool is_safe_relpath(const fs::path &p) 
{
    // 1. Normalize the path (pure string manipulation)
    //    "foo/../bar" becomes "bar"
    //    "../foo"     remains "../foo"
    fs::path normal = p.lexically_normal();

    // 2. Reject absolute paths 
    //    Crucial because: path("/data") / path("/etc") yields "/etc"
    if (normal.is_absolute())
        return false;

    // 3. Reject paths that climb up
    //    If the normalized path starts with "..", it breaks out of the root.
    //    We check the first component using the path iterator.
    if (!normal.empty() && *normal.begin() == "..")
        return false;

    // If we get here, the path is safe (assuming no symlinks on disk)
    return true;
}


// -------------------------------------------------------------------------------------------------
//
// RemoveGuard


RemoveGuard::RemoveGuard(const fs::path &filename_, bool exist_ok)
    : filename(filename_)
{
    if (!exist_ok && file_exists(filename))
        throw runtime_error(string(filename) + ": file already exists (RemoveGuard)");
}


RemoveGuard::~RemoveGuard()
{
    if (committed)
        return;

    try {
        // Note: if file doesn't exist, then remove_file() returns false, rather
        // than throwing an exception. This is okay and we don't print a warning.
        remove_file(filename);
    }
    catch (const fs::filesystem_error &e) {
        // In destructor, don't throw -- just print warning on failure.
        cout << "RemoveGuard warning: " << e.what() << endl;
    }
}


void RemoveGuard::commit()
{
    committed = true;
}


// -------------------------------------------------------------------------------------------------
//
// TmpFileGuard


static fs::path make_tmp_filename(const fs::path &filename)
{
    fs::path temp_path = filename;
    temp_path += ".tmp" + ksgpu::make_random_hex_string(8);
    return temp_path;
}


TmpFileGuard::TmpFileGuard(const fs::path &filename_)
{
    filename = filename_;
    tmp_filename = make_tmp_filename(filename);
}


TmpFileGuard::~TmpFileGuard()
{
    if (committed)
        return;

    try {
        // Note: if file doesn't exist, then remove_file() returns false, rather
        // than throwing an exception. This is okay and we don't print a warning.
        remove_file(tmp_filename);
    }
    catch (const fs::filesystem_error &e) {
        // In destructor, don't throw -- just print warning on failure.
        cout << "TmpFileGuard warning: " << e.what() << endl;
    }
}


void TmpFileGuard::commit()
{
    rename_file(tmp_filename, filename);
    committed = true;
}


// -------------------------------------------------------------------------------------------------
//
// File (RAII wrapper around unix file descriptor, currently only used in hwtest)

    
File::File(const string &filename_, int oflags, int mode)
    : filename(filename_)
{
    fd = open(filename.c_str(), oflags, mode);
    
    if (fd < 0) {
        // FIXME exception text should show 'oflags' and 'mode'.
        stringstream ss;
        ss << filename << ": open() failed: " << strerror(errno);
        throw runtime_error(ss.str());
    }
}

File::~File()
{
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}


void File::write(const void *p, long nbytes)
{
    if (nbytes == 0)
        return;
    
    xassert(p != nullptr);
    xassert(nbytes > 0);
    xassert(fd >= 0);

    // C++ doesn't alllow '+=' on a (const void *).
    const char *pc = reinterpret_cast<const char *> (p);
        
    while (nbytes > 0) {
        long n = ::write(fd, pc, nbytes);
        
        if (n < 0) {
            stringstream ss;
            ss << filename << ": write() failed: " << strerror(errno);
            throw runtime_error(ss.str());
        }
        
        if (n == 0) {
            // Just being paranoid -- I don't think this can actually happen.
            stringstream ss;
            ss << filename << ": write() returned zero?!";
            throw runtime_error(ss.str());
        }
        
        pc += n;
        nbytes -= n;
    }
}


}   // namespace pirate
