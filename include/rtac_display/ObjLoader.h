#ifndef _DEF_RTAC_DISPLAY_OBJ_LOADER_H_
#define _DEF_RTAC_DISPLAY_OBJ_LOADER_H_

#include <iostream>
#include <memory>
#include <vector>

#include <rtac_base/files.h>

#include <rtac_display/GLMesh.h>

namespace rtac { namespace display {

template <typename T>
class Chunk
{
    public:

    using value_type = T;
    using const_iterator = typename std::vector<T>::const_iterator;

    protected:

    std::vector<T> data_;
    unsigned int currentSize_;

    public:

    Chunk(unsigned int capacity = 100) : 
        data_(capacity),
        currentSize_(0)
    {}

    unsigned int capacity() const { return data_.size(); }
    unsigned int size()     const { return currentSize_; }

    void push_back(const T& value) {
        if(currentSize_ >= this->capacity()) {
            throw std::runtime_error("Chunk is full !");
        }
        data_[currentSize_] = value;
        currentSize_++;
    }

    T operator[](unsigned int idx) const { return data_[idx]; }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const   { return data_.begin() + this->size(); }
};

template <typename T>
class ChunkContainer
{
    public:
    
    using value_type = T;

    class ConstIterator
    {
        protected:

        typename std::list<Chunk<T>>::const_iterator listIt_;
        typename std::list<Chunk<T>>::const_iterator listLast_;
        typename Chunk<T>::const_iterator            chunkIt_;
        typename Chunk<T>::const_iterator            chunkEnd_;
        
        public:

        ConstIterator(typename std::list<Chunk<T>>::const_iterator listIt,
                      typename std::list<Chunk<T>>::const_iterator listLast) :
            listIt_(listIt), listLast_(listLast),
            chunkIt_(listIt->begin()), chunkEnd_(listIt->end())
        {}

        ConstIterator(typename std::list<Chunk<T>>::const_iterator listIt,
                      typename std::list<Chunk<T>>::const_iterator listLast,
                      typename Chunk<T>::const_iterator            chunkIt,
                      typename Chunk<T>::const_iterator            chunkEnd) :
            listIt_(listIt), listLast_(listLast), chunkIt_(chunkIt), chunkEnd_(chunkEnd)
        {}

        ConstIterator& operator++() {

            if(chunkIt_ + 1 == chunkEnd_) {
                if(listIt_ == listLast_) {
                    chunkIt_++;
                }
                else {
                    listIt_++;
                    chunkIt_  = listIt_->begin();
                    chunkEnd_ = listIt_->end();
                }
            }
            else if(chunkIt_ != chunkEnd_) {
                chunkIt_++;
            }

            return *this;
        }

        bool operator==(const ConstIterator& other) const {
            return listIt_ == other.listIt_ && chunkIt_ == other.chunkIt_;
        }

        bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }

        const T& operator*() const {
            return *chunkIt_;
        }
    };
    
    protected:

    unsigned int chunkSize_;
    std::list<Chunk<T>> chunks_;

    public:

    ChunkContainer(unsigned int chunkSize = 100) :
        chunkSize_(chunkSize),
        chunks_(1, Chunk<T>(chunkSize_))
    {}

    size_t size() const {
        size_t size = 0;
        for(auto chunk : chunks_) {
            size += chunk.size();
        }
        return size;
    }

    void push_back(T value) {
        if(chunks_.back().size() >= chunkSize_) {
            chunks_.push_back(Chunk<T>(chunkSize_));
        }
        chunks_.back().push_back(value);
    }

    ConstIterator begin() const {
        return ConstIterator(chunks_.begin(), std::prev(chunks_.end()));
    }

    ConstIterator end() const {
        return ConstIterator(std::prev(chunks_.end()), std::prev(chunks_.end()),
                             std::prev(chunks_.end())->end(),
                             std::prev(chunks_.end())->end());
    }

    std::vector<T> to_vector() const
    {
        std::vector<T> out(this->size());
        size_t idx = 0;
        for(auto v : *this) {
            out[idx] = v;
            idx++;
        }

        return out;
    }
};

class ObjLoader
{
    public:

    using Point  = GLMesh::Point;
    using Face   = GLMesh::Face;
    using UV     = GLMesh::UV; 
    using Normal = GLMesh::Normal;

    protected:

    std::string datasetPath_;
    std::string objPath_;
    std::string mtlPath_;

    std::vector<Point>  points_;
    std::vector<UV>     uvs_;
    std::vector<Normal> normals_;

    public:

    ObjLoader(const std::string& datasetPath);

    void load_geometry(unsigned int chunkSize = 100);

    std::string dataset_path() const { return datasetPath_; }
    std::string obj_path()     const { return objPath_; }
    std::string mtl_path()     const { return mtlPath_; }

    const std::vector<Point>&  points()  const { return points_; }
    const std::vector<UV>&     uvs()     const { return uvs_; }
    const std::vector<Normal>& normals() const { return normals_; }
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_OBJ_LOADER_H_
