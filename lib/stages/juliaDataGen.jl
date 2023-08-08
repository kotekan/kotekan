module DataGen

println("Julia: Compiling code");

function fill_buffer(ptr::Ptr{UInt8}, sz::Int64)
    println("Julia: Filling buffer with $sz elements...")
    for i in 1:sz
        unsafe_store!(ptr, i % UInt8, i)
    end
    println("Julia: Done filling buffer.")
end

end
