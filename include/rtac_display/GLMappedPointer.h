#ifndef _DEF_RTAC_DISPLAY_GL_MAPPED_POINTER_H_
#define _DEF_RTAC_DISPLAY_GL_MAPPED_POINTER_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

template <typename PointerT>
class GLMappedPointer
{
    protected:

    GLuint   bufferId_;
    GLenum   target_;
    PointerT ptr_;

    struct read_only_access  { constexpr static const GLenum value = GL_READ_ONLY; };
    struct read_write_access { constexpr static const GLenum value = GL_READ_WRITE; };
    
    public: // temporary for testing

    using Access = typename std::conditional<
        std::is_const<typename std::remove_pointer<PointerT>::type>::value,
        read_only_access, read_write_access>::type;
    
    // only allowing a GLVector to instanciate this class.
    GLMappedPointer(GLuint bufferId, GLenum target = GL_ARRAY_BUFFER);

    public:

    // not allowing copies
    GLMappedPointer(const GLMappedPointer& other) = delete;
    GLMappedPointer& operator=(const GLMappedPointer& other) = delete;
    // allowing move constructor for GLVector to be able to return an instance.
    GLMappedPointer(GLMappedPointer&& other) = default;
    GLMappedPointer& operator=(GLMappedPointer&& other) = default;

    ~GLMappedPointer();

    // removing this should implies that an instance is always valid
    //void unmap() const;
    
    PointerT get() const;
    operator PointerT() const;
};

template <typename PointerT>
GLMappedPointer<PointerT>::GLMappedPointer(GLuint bufferId, GLenum target) :
    bufferId_(bufferId),
    target_(target),
    ptr_(0)
{
    glBindBuffer(target_, bufferId_);
    check_gl("GLMappedPointer : could not bind buffer for mapping.");
    std::cout << GL_READ_ONLY << std::endl;
    std::cout << GL_READ_WRITE << std::endl;
    std::cout << Access::value << std::endl;
    ptr_ = static_cast<PointerT>(glMapBuffer(target_, Access::value));
    check_gl("GLMappedPointer : could map.");
    glBindBuffer(target_, 0);
}

template <typename PointerT>
GLMappedPointer<PointerT>::~GLMappedPointer()
{
    if(!ptr_) return;

    glBindBuffer(target_, bufferId_);
    check_gl("GLMappedPointer : could not bind buffer for unmapping.");
    glUnmapBuffer(target_);
    check_gl("GLMappedPointer : could not  unmap.");
    glBindBuffer(target_, 0);
}

//template <typename PointerT>
//void GLMappedPointer<PointerT>::unmap() const
//{
//    if(!ptr_) return;
//
//    glBindBuffer(target_, bufferId_);
//    check_gl("GLMappedPointer : could not bind buffer for unmapping.");
//    glUnmapBuffer(target_);
//    check_gl("GLMappedPointer : could not  unmap.");
//    glBindBuffer(target_, 0);
//}

template <typename PointerT>
PointerT GLMappedPointer<PointerT>::get() const
{
    if(!ptr_)
        throw std::runtime_error("Resource is unmapped.");
    return ptr_;
}

template <typename PointerT>
GLMappedPointer<PointerT>::operator PointerT() const
{
    return this->get();
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_MAPPED_POINTER_H_
