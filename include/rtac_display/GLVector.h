#ifndef _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
#define _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <rtac_base/types/MappedPointer.h>

#include <rtac_base/cuda/DeviceVector.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <rtac_display/utils.h>
#include <rtac_display/cuda/utils.h>

namespace rtac { namespace display {

template <typename T>
class GLVector
{
    public:

    using value_type      = T;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    protected:

    GLuint bufferId_;
    size_t size_;
    mutable T*     mappedPtr_;

    void allocate(size_t size);
    void free();

    public:

    GLVector();
    GLVector(size_t size);
    GLVector(const GLVector<T>& other);
    GLVector(const std::vector<T>& other);
    ~GLVector();
    
    GLVector& operator=(const GLVector<T>& other);
    GLVector& operator=(const std::vector<T>& other);

    void resize(size_t size);
    size_t size() const;
    size_t capacity() const;

    // not really const but no other way with OpenGL interface
    GLuint gl_id() const;

    // work needed here
    void bind(GLenum target = GL_ARRAY_BUFFER) const; 
    void unbind(GLenum target = GL_ARRAY_BUFFER) const;
    
    // BELOW HERE IS MORE ADVANCED USAGE ////////////////////////
    // (interop with CUDA and host memory mapping functions).
    // BELOW HERE IS MORE ADVANCED USAGE ////////////////////////

    // Mapping Functions for host access (CPU) to device data (GPU-OpenGL)
    protected:

    T*       do_map();
    T*       do_map_write_only();
    const T* do_map() const;

    public:

    using MappedPointer      = rtac::types::MappedPointer<GLVector<T>>;
    using ConstMappedPointer = rtac::types::MappedPointer<const GLVector<T>>;

    MappedPointer      map(bool writeOnly = false);
    ConstMappedPointer map() const;
    void               unmap() const;

    // Mapping Functions for device CUDA access (GPU-CUDA) to device data
    // (GPU-OpenGL).
    protected:

    mutable cudaGraphicsResource* cudaResource_;
    mutable T*                    cudaDevicePtr_;

    const T* do_map_cuda() const;
    T*       do_map_cuda();
    
    public:

    MappedPointer      map_cuda();
    ConstMappedPointer map_cuda() const;
    void unmap_cuda() const;
    
    // CUDA helpers
    GLVector(const rtac::cuda::DeviceVector<T>& other);
    GLVector(const rtac::cuda::HostVector<T>& other);

    GLVector& operator=(const rtac::cuda::DeviceVector<T>& other);
    GLVector& operator=(const rtac::cuda::HostVector<T>& other);

    rtac::cuda::DeviceVector<T>& to_device_vector(rtac::cuda::DeviceVector<T>& other) const;
    rtac::cuda::DeviceVector<T>  to_device_vector() const;
};

// implementation
template <typename T>
GLVector<T>::GLVector() :
    bufferId_(0),
    size_(0),
    mappedPtr_(nullptr),
    cudaResource_(nullptr),
    cudaDevicePtr_(nullptr)
{}

template <typename T>
GLVector<T>::GLVector(size_t size) :
    GLVector()
{
    this->resize(size);
}

template <typename T>
GLVector<T>::GLVector(const GLVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::GLVector(const std::vector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::~GLVector()
{
    this->free();
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const GLVector<T>& other)
{
    this->resize(other.size());

    other.bind(GL_COPY_READ_BUFFER);
    this->bind(GL_COPY_WRITE_BUFFER);

    glCopyBufferSubData(GL_COPY_READ_BUFFER,
                        GL_COPY_WRITE_BUFFER,
                        0, 0, this->size()*sizeof(T));

    this->unbind(GL_COPY_WRITE_BUFFER);
    other.unbind(GL_COPY_READ_BUFFER);

    return *this;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const std::vector<T>& other)
{
    this->resize(other.size());
    this->bind(GL_ARRAY_BUFFER);
    
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), other.data());

    this->unbind(GL_ARRAY_BUFFER);
    return *this;
}

template <typename T>
void GLVector<T>::allocate(size_t size)
{
    if(!bufferId_)
        glGenBuffers(1, &bufferId_);
    this->bind();
    glBufferData(GL_ARRAY_BUFFER, size*sizeof(T), NULL, GL_STATIC_DRAW);
    this->unbind();
}

template <typename T>
void GLVector<T>::free()
{
    if(bufferId_)
        glDeleteBuffers(1, &bufferId_);
    bufferId_ = 0;
    size_     = 0;
}

template <typename T>
void GLVector<T>::resize(size_t size)
{
    if(this->capacity() < size)
        this->allocate(size);
    size_ = size;
}

template <typename T>
size_t GLVector<T>::size() const
{
    return size_;
}

template <typename T>
size_t GLVector<T>::capacity() const
{
    if(!bufferId_)
        return 0;
    GLint capa;
    this->bind();
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &capa);
    this->unbind();
    return capa / sizeof(T);
}

template <typename T>
GLuint GLVector<T>::gl_id() const
{
    return bufferId_;
}

template <typename T>
void GLVector<T>::bind(GLenum target) const
{
    glBindBuffer(target, this->gl_id());
    check_gl("GLVector::bind : could not bind buffer.");
}

template <typename T>
void GLVector<T>::unbind(GLenum target) const
{
    glBindBuffer(target, 0);
    check_gl("GLVector::unbind : could not unbind buffer (invalid target).");
}

// Mapping functions
template <typename T>
T* GLVector<T>::do_map()
{
    if(mappedPtr_)
        throw std::runtime_error("GLVector already mapped. Cannot map a second time.");

    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}

template <typename T>
T* GLVector<T>::do_map_write_only()
{
    if(mappedPtr_)
        throw std::runtime_error("GLVector already mapped. Cannot map a second time.");

    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}


template <typename T>
const T* GLVector<T>::do_map() const
{
    if(mappedPtr_)
        throw std::runtime_error("GLVector already mapped. Cannot map a second time.");

    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}

template <typename T>
typename GLVector<T>::MappedPointer GLVector<T>::map(bool writeOnly)
{
    if(writeOnly) {
        return MappedPointer(this,
                             &GLVector<T>::do_map_write_only,
                             &GLVector<T>::unmap);
    }
    else {
        return MappedPointer(this,
                             &GLVector<T>::do_map,
                             &GLVector<T>::unmap);
    }
}

template <typename T>
typename GLVector<T>::ConstMappedPointer GLVector<T>::map() const
{
    return ConstMappedPointer(this,
                              &GLVector<T>::do_map,
                              &GLVector<T>::unmap);
}

template <typename T>
void GLVector<T>::unmap() const
{
    if(!mappedPtr_)
        throw std::runtime_error(
            "GLVector not mapped. Cannot unmap (this error is sign of a hidden issue)");

    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for unmapping.");
    glUnmapBuffer(GL_ARRAY_BUFFER);
    check_gl("GLVector : could not  unmap.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    mappedPtr_ = nullptr;
}

template <typename T>
const T* GLVector<T>::do_map_cuda() const
{
    if(cudaDevicePtr_)
        throw std::runtime_error("GLVector already mapped on CUDA. Cannot map a second time.");

    CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
        &cudaResource_, bufferId_, cudaGraphicsRegisterFlagsReadOnly) );
    CUDA_CHECK( cudaGraphicsMapResources(1, &cudaResource_) );
    
    size_t accessibleSize = 0;
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&cudaDevicePtr_), &accessibleSize, cudaResource_) );

    if(accessibleSize < this->size()*sizeof(T)) {
        std::ostringstream oss;
        oss << "Discrepancy between mapped size of GL buffer "
            << "and expected buffer size (expected : "
            << this->size()*sizeof(T) << ", got " << accessibleSize
            << "). Cannot map GLVector on CUDA pointer.";
        throw std::runtime_error(oss.str());
    }

    return cudaDevicePtr_;
}

template <typename T>
T* GLVector<T>::do_map_cuda()
{
    if(cudaDevicePtr_)
        throw std::runtime_error("GLVector already mapped on CUDA. Cannot map a second time.");

    CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
        &cudaResource_, bufferId_, cudaGraphicsRegisterFlagsWriteDiscard) );
    CUDA_CHECK( cudaGraphicsMapResources(1, &cudaResource_) );
    
    size_t accessibleSize = 0;
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&cudaDevicePtr_), &accessibleSize, cudaResource_) );

    if(accessibleSize < this->size()*sizeof(T)) {
        std::ostringstream oss;
        oss << "Discrepancy between mapped size of GL buffer "
            << "and expected buffer size (expected : "
            << this->size()*sizeof(T) << ", got " << accessibleSize
            << "). Cannot map GLVector on CUDA pointer.";
        throw std::runtime_error(oss.str());
    }

    return cudaDevicePtr_;
}

template <typename T>
typename GLVector<T>::MappedPointer GLVector<T>::map_cuda()
{
    return MappedPointer(this,
                         &GLVector<T>::do_map_cuda,
                         &GLVector<T>::unmap_cuda);
}

template <typename T>
typename GLVector<T>::ConstMappedPointer GLVector<T>::map_cuda() const
{
    return ConstMappedPointer(this,
                              &GLVector<T>::do_map_cuda,
                              &GLVector<T>::unmap_cuda);
}

template <typename T>
void GLVector<T>::unmap_cuda() const
{
    if(!cudaDevicePtr_)
        throw std::runtime_error(
            "GLVector not CUDA-mapped. Cannot unmap (this error is sign of a hidden issue)");

    CUDA_CHECK( cudaGraphicsUnmapResources(1, &cudaResource_) );
    CUDA_CHECK( cudaGraphicsUnregisterResource(cudaResource_) );
    cudaDevicePtr_ = nullptr;
    cudaResource_  = nullptr;
}

// CUDA helpers implementations
template <typename T>
GLVector<T>::GLVector(const rtac::cuda::DeviceVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::GLVector(const rtac::cuda::HostVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::DeviceVector<T>& other)
{
    this->resize(other.size());
    auto devicePtr = this->map_cuda();
    CUDA_CHECK( cudaMemcpy(devicePtr, other.data(), this->size()*sizeof(T),
                           cudaMemcpyDeviceToDevice) );
    return *this;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::HostVector<T>& other)
{
    this->resize(other.size());
    this->bind(GL_ARRAY_BUFFER);
    
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), other.data());

    this->unbind(GL_ARRAY_BUFFER);
    return *this;
}

template <typename T>
rtac::cuda::DeviceVector<T>&
GLVector<T>::to_device_vector(rtac::cuda::DeviceVector<T>& other) const
{
    other.resize(this->size());
    auto devicePtr = this->map_cuda();
    CUDA_CHECK( cudaMemcpy(other.data(), devicePtr, other.size()*sizeof(T),
                           cudaMemcpyDeviceToDevice) );
    return other;
}

template <typename T>
rtac::cuda::DeviceVector<T> GLVector<T>::to_device_vector() const
{
    rtac::cuda::DeviceVector<T> res(this->size());
    return this->to_device_vector(res);
}

}; //namespace display
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::display::GLVector<T>& v)
{
    os << "(";
    auto data = v.map();
    if(v.size() <= 16) {
        os << data[0];
        for(int i = 1; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    else {
        for(int i = 0; i < 3; i++) {
            os << data[i] << " ";
        }
        os << "...";
        for(int i = v.size() - 3; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    os << ")";
    return os;
}

#endif //_DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
