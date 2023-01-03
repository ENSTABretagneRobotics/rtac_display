#ifndef _DEF_RTAC_DISPLAY_H_
#define _DEF_RTAC_DISPLAY_H_

#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>
#include <tuple>

#include <rtac_base/time.h>
#include <rtac_base/types/CallbackQueue.h>
#include <rtac_base/types/Image.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLFWContext.h>
#include <rtac_display/DrawingSurface.h>
#include <rtac_display/EventHandler.h>

#include <GLFW/glfw3.h>

namespace rtac { namespace display {

/**
 * Main display class. Handles window creation, receives input event, manage
 * Views and Renderers.
 *
 * Display is the very first class to be instanciated before any other display
 * related class. It will open a new window and create the OpenGL context
 * necessary to use the other features of rtac_display. After an instance of
 * Display was created, the user can create and add new views (manage camera)
 * and renderers (make the OpenGL API call to draw a specific object) to the
 * Display object.
 *
 * It will also manage user input events ny providing an API for adding
 * callbacks for mouse and keyboard.
 *
 * Caution : Only one instance of Display at a time is supported for now.
 */
class Display : public DrawingSurface
{
    public:
    
    using Context   = GLFWContext;
    using Window    = Context::Window;
    using Shape     = View::Shape;
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::Ptr>;

    using KeyCallbacks           = rtac::CallbackQueue<int,int,int,int>;
    using MousePositionCallbacks = rtac::CallbackQueue<double,double>;
    using MouseButtonCallbacks   = rtac::CallbackQueue<int,int,int>;
    using ScrollCallbacks        = rtac::CallbackQueue<double,double>;

    using KeyCallbackT           = KeyCallbacks::CallbackT;
    using MousePositionCallbackT = MousePositionCallbacks::CallbackT;
    using MouseButtonCallbackT   = MouseButtonCallbacks::CallbackT;
    using ScrollCallbackT        = ScrollCallbacks::CallbackT;

    protected:
    
    Window       window_;
    Views        views_;
    Renderers    renderers_;
    
    rtac::time::FrameCounter frameCounter_;
    bool displayFrameRate_;

    // Event callback queues
    KeyCallbacks           keyCallbacks_;
    MousePositionCallbacks mousePositionCallbacks_;
    MouseButtonCallbacks   mouseButtonCallbacks_;
    ScrollCallbacks        scrollCallbacks_;

    static std::tuple<Context::Ptr, Window, Shape> create_window_data(
                                                 size_t width, size_t height,
                                                 const std::string& title,
                                                 const Context::Ptr& sharedContext);
    Display(const std::tuple<Context::Ptr, Window, Shape>& windowData);

    public:

    Display(size_t width = 800, size_t height = 600,
            const std::string& title = "rtac_display",
            const Context::Ptr& sharedContext = nullptr);
    Display(const Context::Ptr& sharedContext);
    void terminate();

    Context::Ptr context() const {
        return std::dynamic_pointer_cast<GLFWContext>(this->Renderer::context());
    }

    Window window();
    void grab_context() const;
    void release_context() const;

    Shape window_shape() const;
    int should_close() const;
    int is_drawing();
    void wait_for_close() const;
    
    virtual void draw();
    template <typename T, template<typename> class VectorT>
    void take_screenshot(rtac::Image<T,VectorT>& output);
    //template <typename T>
    //void take_screenshot<GLVector>(rtac::Image<T,GLVector>& output);

    void enable_frame_counter();
    void disable_frame_counter();
    void limit_frame_rate(double fps);
    void free_frame_rate();

    // event related methods
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int modes);
    static void mouse_position_callback(GLFWwindow* window, double x, double y);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double x, double y);

    unsigned int add_key_callback(const KeyCallbackT& callback);
    unsigned int add_mouse_position_callback(const MousePositionCallbackT& callback);
    unsigned int add_mouse_button_callback(const MouseButtonCallbackT& callback);
    unsigned int add_scroll_callback(const ScrollCallbackT& callback);
    
    void add_event_handler(const EventHandler::Ptr& eventHandler);
};

template <typename T, template<typename> class VectorT>
void Display::take_screenshot(rtac::Image<T,VectorT>& output)
{
    this->grab_context();
    auto shape = this->window_shape();
    output.resize({shape.width  - this->viewportOrigin_.x,
                   shape.height - this->viewportOrigin_.y});
    
    glReadPixels(this->viewportOrigin_.x,
                 this->viewportOrigin_.y,
                 shape.width, shape.height,
                 GLFormat<T>::PixelFormat,
                 GLFormat<T>::Type,
                 output.data());
    GL_CHECK_LAST();
}

//template <typename T>
//void Display::take_screenshot<GLVector>(rtac::Image<T,GLVector>& output)
//{
//    auto shape = this->window_shape();
//    output.resize({shape.width  - this->viewportOrigin_.x,
//                   shape.height - this->viewportOrigin_.y});
//
//    output.data().bind(GL_PIXEL_PACK_BUFFER);
//    glReadPixels(this->viewportOrigin_.x,
//                 this->viewportOrigin_.y,
//                 shape.width, shape.height,
//                 GLFormat<T>::PixelFormat,
//                 GLFormat<T>::Type,
//                 output.data().data());
//    output.data().unbind(GL_PIXEL_PACK_BUFFER);
//    GL_CHECK_LAST();
//}

}; //namespace display
}; //namespace rtac


#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
