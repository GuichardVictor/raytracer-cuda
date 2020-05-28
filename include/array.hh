#pragma once

template <class T>
class Array
{
public:
    __device__ __host__ Array(T** _list, size_t _count)
        : list{_list}, count{_count}
    {}

public:
    T** list;
    size_t count;
};