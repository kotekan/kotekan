#ifndef _PIRATE_FILE_UTILS_HPP
#define _PIRATE_FILE_UTILS_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <fcntl.h>  // open flags (O_RDONLY, etc.)

namespace pirate {
#if 0
}  // compiler pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// Wrappers around filesystem operations, that throw exceptions with customized error-text.


// pirate::create_directories() wraps fs::create_directories().
// Returns true if a new directory was created, false otherwise.
extern bool create_directories(const std::filesystem::path &p);

// pirate::remove_file() wraps fs::remove().
// Deletes a file or an empty directory.
// Returns true if file/directory was deleted, false if it didn't exist.
extern bool remove_file(const std::filesystem::path &p);

// pirate::rename() wraps fs::rename().
// Moves or renames a file or directory.
// Note: On POSIX, this is atomic. If 'to' already exists, it is replaced.
extern void rename_file(const std::filesystem::path &from, const std::filesystem::path &to);

// pirate::create_hard_link() wraps fs::create_hard_link().
extern void create_hard_link(const std::filesystem::path &target, const std::filesystem::path &link);

// pirate::file_exists() wraps fs::exists().
extern bool file_exists(const std::filesystem::path &p);

// pirate::copy_file() wraps fs::copy_file().
// Returns true if the file was copied, false otherwise.
//
// Options reference (bitmask):
//   fs::copy_options::none               // Error if dest exists (Default)
//   fs::copy_options::skip_existing      // Do nothing if dest exists
//   fs::copy_options::overwrite_existing // Replace dest if it exists
//   fs::copy_options::update_existing    // Replace only if 'from' is newer than 'to'
//   fs::copy_options::copy_symlinks      // If 'from' is symlink, copy as symlink (not file)
//   fs::copy_options::skip_symlinks      // If 'from' is symlink, ignore it

extern bool copy_file(
    const std::filesystem::path &from, 
    const std::filesystem::path &to, 
    std::filesystem::copy_options options = std::filesystem::copy_options::none
);

// All pathnames received by RPC clients MUST be validated with is_safe_relpath()!
// (See e.g. comment in AssembledFrame.hpp)
//
// Returns true if 'p' is a relative path that stays within its parent 
// after lexical normalization (resolving internal "." and "..").
// Rejects absolute paths and paths starting with ".." (e.g. "../foo").
//
// NOTE: This is a purely syntactic check; it does NOT access the disk 
// and cannot detect traversal via existing symbolic links. This is a
// deliberate decision to trade safety for speed, since we're in an HPC
// environment with trusted clients/servers (but want to guard against
// unintentional misbehavior).

extern bool is_safe_relpath(const std::filesystem::path &p);


// -------------------------------------------------------------------------------------------------
//
// RemoveGuard: if the destructor is called before commit(), then call remove_file().
//
// Used RAII-style to ensure that if an exception is thrown during file creation logic,
// then the partially-written file is cleaned up.


struct RemoveGuard
{
    std::filesystem::path filename;
    bool committed = false;
    
    // If exist_ok is false, then the constructor throws an exception if file already exists.
    RemoveGuard(const std::filesystem::path &filename, bool exist_ok=false);
    ~RemoveGuard();
    
    void commit();

    RemoveGuard(const RemoveGuard &) = delete;
    RemoveGuard &operator=(const RemoveGuard &) = delete;
};


// -------------------------------------------------------------------------------------------------
//
// TmpFileGuard: constructor initializes 'tmp_filename' to {filename}.tmp{RANDSTRING}.
// commit() renames tmp_filename -> filename, by calling rename().
// If the destructor is called before commit(), then unlink() the tmp file.
// Used to ensure that if an exception/crash happens during file creation logic, then the
// partially-written file is cleaned up if possible, or persists with a tmp filename if not.


struct TmpFileGuard
{
    std::filesystem::path filename;
    std::filesystem::path tmp_filename;
    bool committed = false;
    
    TmpFileGuard(const std::filesystem::path &filename);
    ~TmpFileGuard();
    
    void commit();

    TmpFileGuard(const TmpFileGuard &) = delete;
    TmpFileGuard &operator=(const TmpFileGuard &) = delete;
};


// -------------------------------------------------------------------------------------------------
//
// File: RAII wrapper for unix file descriptor. (Currently only used in hwtest.)


struct File
{
    std::string filename;
    int fd = -1;
    
    // Constructor opens file.
    // Suggest oflags = (O_RDONLY) for reading, and (O_WRONLY | O_CREAT | O_TRUNC) for writing.
    
    File(const std::string &filename, int oflags, int mode=0644);
    ~File();

    void write(const void *p, long nbytes);
    
    // The File class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<File> fp = make_shared<File> (filename, oflags, mode);
    
    File(const File &) = delete;
    File &operator=(const File &) = delete;
};


} // namespace pirate

#endif  // _PIRATE_FILE_UTILS_HPP
