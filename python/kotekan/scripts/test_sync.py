# Python program to explain os.fsync() method
   
# importing os module
import os
import io 
import h5py
 
# File path
path = 'file.txt'
 
# Open the file and get
# the file descriptor
# associated with
# using os.open() method
f = open(path, 'a')
fd = f.fileno()
#fd = os.open(path, os.O_RDWR)
print(fd) 
 
# Write a bytestring
str = "GeeksforGeeks"
 
f.write(str)
#os.write(fd, str)
 
 
# The written string is
# available in program buffer
# but it might not actually
# written to disk until
# program is closed or
# file descriptor is closed.
 
# sync. all internal buffers
# associated with the file descriptor
# with disk (force write of file)
# using os.fsync() method
print(type(fd))
os.fsync(fd)
print("Force write of file committed successfully")
 
# Close the file descriptor
f.close()
#os.close(fd)


with open(path) as f:
    print(f.read())
    os.fsync(f.fileno())

def raw_baseband_frames(file_name: str, buf: bytes):
    """Iterates over frames in a raw baseband file"""
    with io.FileIO(file_name, "rb") as raw_file:
        while raw_file.readinto(buf):
            yield buf
        size = os.path.getsize(file_name)
        os.fsync(raw_file.fileno())
        os.posix_fadvise(raw_file.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
        print("DONE")
file_name = "/data/baseband_raw/baseband_raw_20211022092827/baseband_20211022092827_9.data"
buf = bytearray(1048576+96)

c = 0
for b in raw_baseband_frames(file_name, buf):
    #print(c)
    c += 1

pid = os.getpid()
file_name = "/data/chime/baseband/raw/2021/10/20/astro_20211020204543/baseband_20211020204543_9.h5" 
f = h5py.File(file_name)
f_fd = f.id.get_vfd_handle()
size = os.path.getsize(file_name)
os.fsync(f_fd)
os.posix_fadvise(f_fd, 0, size, os.POSIX_FADV_DONTNEED)
f.close()
print("DONENNN")
