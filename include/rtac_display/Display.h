#ifndef _DEF_RTAC_DISPLAY_H_
#define _DEF_RTAC_DISPLAY_H_

#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

//#include <glm/glm.hpp>
//#include <glm/gtx/transform.hpp>
//#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include <rtac_base/time.h>
#include <rtac_base/types/CallbackQueue.h>

#include <rtac_display/utils.h>
#include <rtac_display/DrawingSurface.h>
#include <rtac_display/EventHandler.h>

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

    using Window    = std::shared_ptr<GLFWwindow>;
    using Shape     = View::Shape;
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::Ptr>;

    using KeyCallbacks           = rtac::types::CallbackQueue<int,int,int,int>;
    using MousePositionCallbacks = rtac::types::CallbackQueue<double,double>;
    using MouseButtonCallbacks   = rtac::types::CallbackQueue<int,int,int>;
    using ScrollCallbacks        = rtac::types::CallbackQueue<double,double>;

    using KeyCallbackT           = KeyCallbacks::CallbackT;
    using MousePositionCallbackT = MousePositionCallbacks::CallbackT;
    using MouseButtonCallbackT   = MouseButtonCallbacks::CallbackT;
    using ScrollCallbackT        = ScrollCallbacks::CallbackT;

    protected:
    
    Window       window_;
    Views        views_;
    Renderers    renderers_;
    EventHandler eventHandler_;
    
    rtac::time::FrameCounter frameCounter_;
    bool frameCounterEnabled_;

    // Event callback queues
    KeyCallbacks           keyCallbacks_;
    MousePositionCallbacks mousePositionCallbacks_;
    MouseButtonCallbacks   mouseButtonCallbacks_;
    ScrollCallbacks        scrollCallbacks_;

    public:

    Display(size_t width = 800, size_t height = 600,
            const std::string& title = "rtac_display",
            const Window& sharedContext = nullptr);
    Display(const Window& sharedContext);
    void terminate();

    Window window();
    void make_current();

    Shape window_shape() const;
    int should_close() const;
    int is_drawing();
    void wait_for_close() const;
    
    virtual void draw();

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

}; //namespace display
}; //namespace rtac


#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
