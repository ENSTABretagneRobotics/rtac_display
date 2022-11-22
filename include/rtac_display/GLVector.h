#ifndef _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
#define _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <rtac_base/types/MappedPointer.h>

#ifdef RTAC_DISPLAY_CUDA // CUDA support is optional
    #include <cuda_runtime.h>
    #include <cuda_gl_interop.h>
    #include <rtac_base/cuda/DeviceVector.h>
    #include <rtac_base/cuda/HostVector.h>
#endif

#include <rtac_display/utils.h>

namespace rtac { namespace display {

/**
 * Helper class to manage OpenGL [buffer
 * Objects](https://www.khronos.org/opengl/wiki/Buffer_Object) as a typed
 * vector.
 *
 * OpenGL buffer objects are untyped chunks of GPU memory which can be
 * cumbersome to use. This object provides an easier memory management.
 * GLVectors are typed and can be resized and copied in a similar way than the
 * C++ standard library containers. 
 *
 * However, it does not provide direct element access and iterators since the
 * data is stored on the GPU memory. Element access or modification is perform
 * either by copying a full data vector to GPU memory or by mapping the
 * GLVector data on host memory ("CPU memory").
 *
 * @tparam T Data element type of GLVector. Usually a simple scalar or vector
 *           type.
 */
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
    void clear();

    public:

    GLVector();
    GLVector(size_t size, const T* data = nullptr);
    GLVector(const GLVector<T>& other);
    GLVector(GLVector<T>&& other);
    GLVector(const std::vector<T>& other);
    ~GLVector();
    
    GLVector& operator=(const GLVector<T>& other);
    GLVector& operator=(GLVector<T>&& other);
    GLVector& operator=(const std::vector<T>& other);
    void set_data(unsigned int size, const T* data);
    
    template <template <typename> class VectorT>
    void copy_to(VectorT<T>& other) const;
    void copy_to(T* dst) const;

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

    using MappedPointer      = rtac::MappedPointer<GLVector<T>>;
    using ConstMappedPointer = rtac::MappedPointer<const GLVector<T>>;

    MappedPointer      map(bool writeOnly = false);
    ConstMappedPointer map() const;
    void               unmap() const;
    
    #ifdef RTAC_DISPLAY_CUDA
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
    void set_device_data(unsigned int size, const T* data);

    rtac::cuda::DeviceVector<T>& to_device_vector(rtac::cuda::DeviceVector<T>& other) const;
    rtac::cuda::DeviceVector<T>  to_device_vector() const;

    #endif
};

/**
 * Instanciate a new GLVector.
 *
 * No operation is made on the device, so this can be called before the
 * creation of an OpenGL context. However, no data can be allocated until an
 * OpenGL context has been created.
 */
template <typename T>
GLVector<T>::GLVector() :
    bufferId_(0),
    size_(0),
    mappedPtr_(nullptr)
   
#ifdef RTAC_DISPLAY_CUDA
    ,
    cudaResource_(nullptr),
    cudaDevicePtr_(nullptr)
#endif
{}

/**
 * Instanciate a new GLVector and allocate data on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param size Number of elements to allocate.
 */
template <typename T>
GLVector<T>::GLVector(size_t size, const T* data) :
    GLVector()
{
    this->resize(size);
    if(data)
        this->set_data(size, data);
}

/**
 * Instanciate a new GLVector, allocate data on the device and copy data from
 * another GLVector. Copy happen solely on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other GLVector to be copied.
 */
template <typename T>
GLVector<T>::GLVector(const GLVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

/**
 * Instanciate a new GLVector, allocate data on the device and move data from
 * another GLVector. No copy happens.
 *
 * @param other GLVector to be moved.
 */
template <typename T>
GLVector<T>::GLVector(GLVector<T>&& other) :
    GLVector()
{
    *this = std::move(other);
}

/**
 * Instanciate a new GLVector, allocate data on the device and transfert data
 * from the host to the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other std::vector with data to be copied to the host.
 */
template <typename T>
GLVector<T>::GLVector(const std::vector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::~GLVector()
{
    this->clear();
}

/**
 * Reallocate data if needed and copy an existing GLVector. Copy happen solely
 * on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other GLVector to be copied.
 *
 * @return A reference to this instance.
 */
template <typename T>
GLVector<T>& GLVector<T>::operator=(const GLVector<T>& other)
{
    if(other.size() == 0) return *this;

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

/**
 * Reallocate data if needed and move an existing GLVector. No copy happen.
 *
 * @param other GLVector to be moved.
 *
 * @return A reference to this instance.
 */
template <typename T>
GLVector<T>& GLVector<T>::operator=(GLVector<T>&& other)
{
    bufferId_ = std::exchange(other.bufferId_, bufferId_);
    size_     = std::exchange(other.size_,     size_);
    return *this;
}

/**
 * Reallocate data if needed and copy data from an std::vector. Data is copied
 * from the host to the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other std::vector to be copied.
 *
 * @return A reference to this instance.
 */
template <typename T>
GLVector<T>& GLVector<T>::operator=(const std::vector<T>& other)
{
    this->set_data(other.size(), other.data());
    return *this;
}

/**
 * Reallocate data if needed and copy data from host memory.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param size number of elements to copy.
 * @param data Host memory pointer to the data to be copied.
 */
template <typename T>
void GLVector<T>::set_data(unsigned int size, const T* data)
{
    if(size == 0) return;

    this->resize(size);

    this->bind(GL_ARRAY_BUFFER);
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), data);
    this->unbind(GL_ARRAY_BUFFER);
}

/**
 * Copy data to client memory (host memory in CUDA terminology)
 *
 * @param other a std::vector compliant container.
 */
template <typename T> template <template <typename> class VectorT>
void GLVector<T>::copy_to(VectorT<T>& other) const
{
    other.resize(this->size());
    this->copy_to(other.data());
}

/**
 * Copy data to client memory (host memory in CUDA terminology)
 *
 * @param dst client size memory buffer already allocated.
 */
template <typename T>
void GLVector<T>::copy_to(T* dst) const
{
    if(this->size() == 0) return;

    this->bind(GL_COPY_READ_BUFFER);

    glGetBufferSubData(GL_COPY_READ_BUFFER, 0,
                        this->size()*sizeof(T), dst);

    this->unbind(GL_COPY_READ_BUFFER);
}

/**
 * Reallocate data on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param size Number of elements to allocate.
 */
template <typename T>
void GLVector<T>::allocate(size_t size)
{
    if(!bufferId_)
        glGenBuffers(1, &bufferId_);
    this->bind();
    glBufferData(GL_ARRAY_BUFFER, size*sizeof(T), NULL, GL_STATIC_DRAW);
    this->unbind();
}

/**
 * Free data on the device. It is safe to call this function several time in a
 * row.
 *
 * An OpenGL context must have been created beforehand.
 */
template <typename T>
void GLVector<T>::clear()
{
    if(bufferId_)
        glDeleteBuffers(1, &bufferId_);
    bufferId_ = 0;
    size_     = 0;
}

/**
 * Resize data on the device. Reallocation happen only if requested data is
 * larger than already allocated data.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param size Number of elements to allocate.
 */
template <typename T>
void GLVector<T>::resize(size_t size)
{
    if(this->capacity() < size)
        this->allocate(size);
    size_ = size;
}

/**
 * @return current size of the vector. (Allocated size might be larger).
 */
template <typename T>
size_t GLVector<T>::size() const
{
    return size_;
}

/**
 * @return currently allocated size on the device in number of elements.
 */
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

/**
 * @return the name of the OpenGL buffer object. (can be used in glBindBuffer).
 */
template <typename T>
GLuint GLVector<T>::gl_id() const
{
    return bufferId_;
}

/**
 * Bind the buffer to an OpenGL target (such as GL_ARRAY_BUFFER).
 */
template <typename T>
void GLVector<T>::bind(GLenum target) const
{
    glBindBuffer(target, this->gl_id());
}

/**
 * Unbind all buffer from an OpenGL target.
 *
 * This was implemented as a method in this class to keep a symmetry in the
 * interface, but it is not necessary. Might be removed in the future.
 */
template <typename T>
void GLVector<T>::unbind(GLenum target) const
{
    glBindBuffer(target, 0);
    //check_gl("GLVector::unbind : could not unbind buffer (invalid target).");
}

/**
 * Map device memory of the GLVector to host memory for its data to be
 * read from/written to by the host.
 *
 * The GLVector must be unmapped before use in OpenGL API calls. Otherwise,
 * OpenGL error will be generated.
 *
 * @return a pointer to host memory which can be read or written to by the
 *         host.
 */
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

/**
 * Map device memory of the GLVector to host memory for its data to be
 * written to by the host.
 *
 * The GLVector must be unmapped before use in OpenGL API calls. Otherwise,
 * OpenGL error will be generated.
 *
 * @return a pointer to host memory on which host can write to. Data is write
 *         only.
 */
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

/**
 * Map device memory of the GLVector to host memory for its data to be
 * read by the host.
 *
 * The GLVector must be unmapped before use in OpenGL API calls. Otherwise,
 * OpenGL error will be generated.
 *
 * @return a pointer to host memory to be read by the host. Data is read only.
 */
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

/**
 * Map GLVector data to host and return a MappedPointer. The MappedPointer will
 * automatically unmap the GLVector on destruction.
 *
 * The GLVector cannot be used in OpenGL API calls until this MappedPointer is
 * destroyed and the GLVector unmapped.
 *
 * @param writeOnly If writeOnly is set to true, the Mapped Pointer can only be
 *                  written to. If writeOnly is false, data can be read and
 *                  written. Specifying writeOnly when needed might accelerate
 *                  writing.
 * @return a MappedPointer to the GLVector device data.
 */
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

/**
 * Map GLVector data to host and return a MappedPointer. The MappedPointer will
 * automatically unmap the GLVector on destruction. MappedPointer is read-only.
 *
 * The GLVector cannot be used in OpenGL API calls until this MappedPointer is
 * destroyed and the GLVector unmapped.
 *
 * @return a read-only MappedPointer to the GLVector device data.
 */
template <typename T>
typename GLVector<T>::ConstMappedPointer GLVector<T>::map() const
{
    return ConstMappedPointer(this,
                              &GLVector<T>::do_map,
                              &GLVector<T>::unmap);
}

/**
 * Unmap GLVector. This will update device with the data which was written to
 * the mapping location.
 *
 * It is mandatory to unmap a buffer before use on OpenGL API call.
 *
 * If the GLVector was mapped with GLVector::map, the resulting MappedPointer
 * will automatically unmap the GLVector by calling this function when it is
 * destroyed.
 */
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

// BELOW HERE ARE CUDA SPECIFIC FUNCTIONALITIES
#ifdef RTAC_DISPLAY_CUDA
/**
 * Map device memory of the GLVector to a read-only CUDA device pointer.
 *
 * This allows to use OpenGL Buffer Object data directly into CUDA API calls.
 * It is helpful to perform GPU treatment on displayed data before rendering.
 *
 * The GLVector must be unmapped before use in OpenGL API calls. Otherwise,
 * OpenGL error will be generated.
 *
 * @return a CUDA const pointer to device memory.
 */
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

/**
 * Map device memory of the GLVector to a read-write CUDA device pointer.
 *
 * This allows to use OpenGL Buffer Object data directly into CUDA API calls.
 * It is helpful to perform GPU treatment on displayed data before rendering.
 *
 * The GLVector must be unmapped before use in OpenGL API calls. Otherwise,
 * OpenGL error will be generated.
 *
 * @return a CUDA pointer to device memory.
 */
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

/**
 * Map GLVector data to CUDA device and return a MappedPointer. The
 * MappedPointer will automatically unmap the GLVector on destruction.
 *
 * This allows to use OpenGL Buffer Object data directly into CUDA API calls.
 * It is helpful to perform GPU treatment on displayed data before rendering.
 *
 * The GLVector cannot be used in OpenGL API calls until this MappedPointer is
 * destroyed and the GLVector unmapped.
 *
 * @return a read-write MappedPointer to CUDA device memory.
 */
template <typename T>
typename GLVector<T>::MappedPointer GLVector<T>::map_cuda()
{
    return MappedPointer(this,
                         &GLVector<T>::do_map_cuda,
                         &GLVector<T>::unmap_cuda);
}

/**
 * Map GLVector data to CUDA device and return a MappedPointer. The
 * MappedPointer will automatically unmap the GLVector on destruction. The
 * MappedPointer is read-only.
 *
 * This allows to use OpenGL Buffer Object data directly into CUDA API calls.
 * It is helpful to perform GPU treatment on displayed data before rendering.
 *
 * The GLVector cannot be used in OpenGL API calls until this MappedPointer is
 * destroyed and the GLVector unmapped.
 *
 * @return a read-write MappedPointer to CUDA device memory.
 */
template <typename T>
typename GLVector<T>::ConstMappedPointer GLVector<T>::map_cuda() const
{
    return ConstMappedPointer(this,
                              &GLVector<T>::do_map_cuda,
                              &GLVector<T>::unmap_cuda);
}

/**
 * Unmap GLVector.
 *
 * It is mandatory to unmap a buffer before use on OpenGL API call.
 *
 * If the GLVector was mapped with GLVector::map_cuda, the resulting MappedPointer
 * will automatically unmap the GLVector by calling this function when it is
 * destroyed.
 */
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
/**
 * Instanciate a new GLVector, allocate data on the device and copy data from
 * a rtac::cuda::DeviceVector. Copy happen solely on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other DeviceVector to be copied.
 */
template <typename T>
GLVector<T>::GLVector(const rtac::cuda::DeviceVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

/**
 * Instanciate a new GLVector, allocate data on the device and copy data from
 * a rtac::cuda::HostVector. Copy happen from the host to the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other HostVector to be copied.
 */
template <typename T>
GLVector<T>::GLVector(const rtac::cuda::HostVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

/**
 * Reallocate data if needed and copy from a rtac::cuda::DeviceVector. Copy
 * happen solely on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other DeviceVector to be copied.
 *
 * @return A reference to this instance.
 */
template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::DeviceVector<T>& other)
{
    if(other.size() == 0) return *this;

    this->resize(other.size());
    auto devicePtr = this->map_cuda();
    CUDA_CHECK( cudaMemcpy(devicePtr, other.data(), this->size()*sizeof(T),
                           cudaMemcpyDeviceToDevice) );
    return *this;
}

/**
 * Reallocate data if needed and copy from a rtac::cuda::HostVector. Copy
 * happen from the host to the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other HostVector to be copied.
 *
 * @return A reference to this instance.
 */
template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::HostVector<T>& other)
{
    if(other.size() == 0) return *this;

    this->resize(other.size());
    this->bind(GL_ARRAY_BUFFER);
    
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), other.data());

    this->unbind(GL_ARRAY_BUFFER);
    return *this;
}

/**
 * Reallocate data if needed and copy from a cuda device pointer. Copy happen
 * solely on the device.
 *
 * An OpenGL context must have been created beforehand.
 *
 * @param other DeviceVector to be copied.
 *
 * @return A reference to this instance.
 */
template <typename T>
void GLVector<T>::set_device_data(unsigned int size, const T* data)
{
    if(size == 0) return;

    this->resize(size);
    auto devicePtr = this->map_cuda();
    CUDA_CHECK( cudaMemcpy(devicePtr, data, this->size()*sizeof(T),
                           cudaMemcpyDeviceToDevice) );
}

/**
 * Copy data to an existing rtac::cuda::DeviceVector. Copy happen solely on
 * the device.
 *
 * @param other a non-const reference to a DeviceVector.
 *
 * @return a reference to the same DeviceVector other.
 */
template <typename T>
rtac::cuda::DeviceVector<T>&
GLVector<T>::to_device_vector(rtac::cuda::DeviceVector<T>& other) const
{
    if(this->size() == 0) return other;

    other.resize(this->size());
    auto devicePtr = this->map_cuda();
    CUDA_CHECK( cudaMemcpy(other.data(), devicePtr, other.size()*sizeof(T),
                           cudaMemcpyDeviceToDevice) );
    return other;
}

/**
 * Creates a new rtac::cuda::DeviceVector and copy data to it. Copy happen
 * solely on the device.
 *
 * @return a reference to the newly created DeviceVector.
 */
template <typename T>
rtac::cuda::DeviceVector<T> GLVector<T>::to_device_vector() const
{
    rtac::cuda::DeviceVector<T> res(this->size());
    return this->to_device_vector(res);
}
#endif

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
