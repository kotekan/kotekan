# module DataGen

println("Julia: Compiling code");

function fill_buffer(ptr::Ptr{UInt8}, sz::Int64)
    println("Julia: Filling buffer");
    for i in 1:sz
        unsafe_store!(p, i % UInt8, i)
    end
end

# end
