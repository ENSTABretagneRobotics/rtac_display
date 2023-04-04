#include <rtac_display/Display.h>

#include <thread>

namespace rtac { namespace display {

std::tuple<GLFWContext::Ptr, Display::Window, Display::Shape> Display::create_window_data(
                                        size_t width, size_t height,
                                        const std::string& title,
                                        const Context::Ptr& sharedContext)
{
    if(!glfwInit()) {
        throw std::runtime_error("GLFW initialization failure.");
    }

    Window otherWindow;
    if(sharedContext)
        otherWindow = sharedContext->window();

    auto newWindow = Window(glfwCreateWindow(width, height,
                                             title.c_str(),
                                             NULL, otherWindow.get()),
                                             glfwDestroyWindow); //custom deleter
    if(!newWindow) {
        const char* desc;
        int code = glfwGetError(&desc);
        std::ostringstream oss;
        oss << "GLFW creation failure (code:"
            << std::hex << code << ") : " << desc;
        throw std::runtime_error(oss.str());
    }

    if(sharedContext)
        return std::make_tuple(sharedContext, newWindow, Shape({width, height}));
    else
        return std::make_tuple(GLFWContext::Create(newWindow), newWindow, Shape({width, height}));
}

Display::Display(const std::tuple<Context::Ptr, Window, Shape>& windowData) :
    DrawingSurface(std::get<0>(windowData)),
    window_(std::get<1>(windowData)),
    displayFrameRate_(false),
    pauseHandler_(samples::PauseHandler::Create())
{
    if(!window_) {
        throw std::runtime_error("Initialization failure");
    }

    this->grab_context();
    //glfwMakeContextCurrent(window_.get());
    
    // init glew (no gl function availabel if not done)
    GLenum initGlewStatus(glewInit());
    if(initGlewStatus != GLEW_OK)
        std::cout << "Failed to initialize glew" << std::endl;
    //std::cout << glGetString(GL_VERSION) << std::endl;
    //if(GLEW_ARB_compute_shader)
    //    std::cout << "Compute shader ok !" << std::endl;

    glClearColor(0.0,0.0,0.0,1.0);
    //glClearColor(0.7,0.7,0.7,1.0);

    // to measure fps
    glfwSwapInterval(0);

    auto width  = std::get<2>(windowData).width;
    auto height = std::get<2>(windowData).height;

    glViewport(0.0,0.0,width,height);

    // Making this the user pointer for callback related features.
    GLFW_CHECK( glfwSetWindowUserPointer(window_.get(), this) );

    this->add_display_flags( DrawingSurface::CLEAR_COLOR 
                           | DrawingSurface::CLEAR_DEPTH);

    this->limit_frame_rate(60.0);

    this->add_event_handler(pauseHandler_);
}

Display::Display(size_t width, size_t height, const std::string& title,
                 const Context::Ptr& sharedContext) :
    Display(Display::create_window_data(width, height, title, sharedContext))
{}

Display::Display(const Context::Ptr& sharedContext) :
    Display(800, 600, "rtac_display", sharedContext)
{}

void Display::terminate()
{
    glfwTerminate();
}

Display::Window Display::window()
{
    return window_;
}

void Display::grab_context() const
{
    //this->context()->make_current();
    glfwMakeContextCurrent(window_.get());
}

void Display::release_context() const
{
    //this->context()->make_current();
    glfwMakeContextCurrent(window_.get());
}

/**
 * @return width and height of the window in a Shape object.
 */
Display::Shape Display::window_shape() const
{
    Shape wSize;
    int width, height;
    glfwGetWindowSize(window_.get(), &width, &height);
    wSize.width  = width;
    wSize.height = height;
    return wSize;
}

/**
 * Check if the Display window should close.
 *
 * This is usually put as the condition in a while loop. While this function
 * returns false, the drawing loop should continue. The window will report true
 * if a closing condition is met (ctrl-c, window close button was clicked...)
 *
 * This function will also handle polled events when called.
 *
 * @return Boolean true if window closing was requested.
 */
int Display::should_close() const
{
    using namespace std;
    glfwPollEvents();
    while(pauseHandler_->is_paused()) {
        glfwPollEvents();
        std::this_thread::sleep_for(100ms);
    }
    return glfwWindowShouldClose(window_.get()) > 0;
}

/**
 * Check if the Display window should close (same as should close, but will
 * also draw the scene).
 *
 * This is usually put as the condition in a while loop. While this function
 * returns false, the drawing loop should continue. The window will report true
 * if a closing condition is met (ctrl-c, window close button was clicked...)
 *
 * This function will also handle polled events when called and draw the scene.
 *
 * @return Boolean true if window closing was requested.
 */
int Display::is_drawing()
{
    this->draw();
    glfwPollEvents();
    return glfwWindowShouldClose(window_.get()) == 0;
}

/**
 * Wait until a window closing condition is met (ctrl-c, window close button
 * clicked..)
 */
void Display::wait_for_close() const
{
    using namespace std;
    while(!this->should_close()) {
        std::this_thread::sleep_for(100ms);
    }
}

/**
 * Update all handled views with the current window size and draw all the
 * handled renderers after clearing the window.
 *
 * If the frame counter is enabled, it will display the frame rate in the
 * standard output.
 */
void Display::draw()
{
    //glfwMakeContextCurrent(window_.get());
    this->grab_context();
    
    auto view = View::Create();
    view->set_screen_size(this->window_shape());
    this->DrawingSurface::draw(view);

    glfwSwapBuffers(window_.get());

    if(displayFrameRate_) {
        std::cout << frameCounter_;
    }
    else {
        frameCounter_.get();
    }
}

/**
 * Enable frame counter and display frame rate in standard output.
 */
void Display::enable_frame_counter()
{
    displayFrameRate_ = true;
}

void Display::disable_frame_counter()
{
    displayFrameRate_ = false;
}

/**
 * Enable frame limiter and set it to a specific frame rate (does no work
 * accurately).
 */
void Display::limit_frame_rate(double fps)
{
    frameCounter_.limit_frame_rate(fps);
}

/**
 * Disable frame limiter and display as fast as possible.
 */
void Display::free_frame_rate()
{
    frameCounter_.free_frame_rate();
}

/**
 * Main keyboard key callback. This will call all users registered key callbacks.
 *
 * Users should not call this directly.
 */
void Display::key_callback(GLFWwindow* window, int key, int scancode, int action, int modes)
{
    auto display = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    if(!display) {
        std::cerr << "Got event but user data not set in GLFWwindow. "
                  << "Cannot transfer events" << std::endl;
        return;
    }
    display->keyCallbacks_.call(key, scancode, action, modes);
}

/**
 * Main mouse position callback. This will call all users registered mouse
 * position callbacks.
 *
 * Users should not call this directly.
 */
void Display::mouse_position_callback(GLFWwindow* window, double x, double y)
{
    auto display = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    if(!display) {
        std::cerr << "Got event but user data not set in GLFWwindow. "
                  << "Cannot transfer events" << std::endl;
        return;
    }
    display->mousePositionCallbacks_.call(x, y);
}

/**
 * Main mouse button callback. This will call all users registered mouse button
 * callbacks.
 *
 * Users should not call this directly.
 */
void Display::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    auto display = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    if(!display) {
        std::cerr << "Got event but user data not set in GLFWwindow. "
                  << "Cannot transfer events" << std::endl;
        return;
    }
    display->mouseButtonCallbacks_.call(button, action, mods);
}

/**
 * Main scroll callback. This will call all users registered scroll callbacks.
 *
 * Users should not call this directly.
 */
void Display::scroll_callback(GLFWwindow* window, double x, double y)
{
    auto display = reinterpret_cast<Display*>(glfwGetWindowUserPointer(window));
    if(!display) {
        std::cerr << "Got event but user data not set in GLFWwindow. "
                  << "Cannot transfer events" << std::endl;
        return;
    }
    display->scrollCallbacks_.call(x, y);
}

/**
 * Add a keyboard key callback.
 *
 * The signature of the callback should be (int,int,int,int). Parameters are :
 * - int key      : keyboard key (value may depend on system configuration,
 *                  i.e. QWERTY, AZERTY...)
 * - int scancode : platform-specific. Constant unique identifier for a key.
 *                  Constant regardless of system configuration, may/will vary
 *                  across platforms.
 * - int action   : Either GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT.
 * - int mods     : Modifier bits (CONTROL, SHIFT, ALT, SUPER ...)
 *
 * See [GLFW input guide](https://www.glfw.org/docs/3.3/input_guide.html) for
 * more information.
 *
 * @param callback Either a function pointer with the signature
 *                 (int,int,int,int) or an equivalent std::function (use a
 *                 std::function to bind a class method to an object with the
 *                 function std::bind. See rtac_base::CallbackQueue,
 *                 or Display::add_event_handler for more information).
 */
unsigned int Display::add_key_callback(const KeyCallbackT& callback)
{
    // Adding all callback handler for now (dynamic ones later)
    GLFW_CHECK( glfwSetKeyCallback(window_.get(), &Display::key_callback) );
    return keyCallbacks_.add_callback(callback);
}

/**
 * Add a mouse position callback.
 *
 * The signature of the callback should be (double, double). Parameters are :
 * - double x, double y : Cursor position measured in screen relative to
 *                        the top-left corner of the window content area. If
 *                        platform supports it, sub-pixel cursor position is
 *                        passed.
 *
 * See [GLFW input guide](https://www.glfw.org/docs/3.3/input_guide.html) for
 * more information.
 *
 * @param callback Either a function pointer with the signature (double,double)
 *                 or an equivalent std::function (use a std::function to bind
 *                 a class method to an object with the function std::bind. See
 *                 rtac_base::CallbackQueue, or
 *                 Display::add_event_handler for more information).
 */
unsigned int Display::add_mouse_position_callback(const MousePositionCallbackT& callback)
{
    GLFW_CHECK( glfwSetCursorPosCallback(window_.get(), &Display::mouse_position_callback) );
    return mousePositionCallbacks_.add_callback(callback);
}

/**
 * Add a mouse button callback.
 *
 * The signature of the callback should be (int,int,int). Parameters are :
 * - int button   : Mouse button id, GLFW_MOUSE_BUTTON_(LEFT/RIGHT/MIDDLE/others)
 * - int action   : Either GLFW_PRESS, GLFW_RELEASE.
 * - int mods     : Modifier bits (CONTROL, SHIFT, ALT, SUPER ...)
 *
 * See [GLFW input guide](https://www.glfw.org/docs/3.3/input_guide.html) for
 * more information.
 *
 * @param callback Either a function pointer with the signature (int,int,int)
 *                 or an equivalent std::function (use a std::function to bind
 *                 a class method to an object with the function std::bind. See
 *                 rtac_base::CallbackQueue, or
 *                 Display::add_event_handler for more information).
 */
unsigned int Display::add_mouse_button_callback(const MouseButtonCallbackT& callback)
{
    GLFW_CHECK( glfwSetMouseButtonCallback(window_.get(), &Display::mouse_button_callback) );
    return mouseButtonCallbacks_.add_callback(callback);
}

/**
 * Add a scroll callback.
 *
 * The signature of the callback should be (double, double). Parameters are :
 * - double x, double y : Scroll offset (mouse scroll is on y axis).
 *
 * See [GLFW input guide](https://www.glfw.org/docs/3.3/input_guide.html) for
 * more information.
 *
 * @param callback Either a function pointer with the signature (double,double)
 *                 or an equivalent std::function (use a std::function to bind
 *                 a class method to an object with the function std::bind. See
 *                 rtac_base::CallbackQueue, or
 *                 Display::add_event_handler for more information).
 */
unsigned int Display::add_scroll_callback(const ScrollCallbackT& callback)
{
    GLFW_CHECK( glfwSetScrollCallback(window_.get(), &Display::scroll_callback) );
    return scrollCallbacks_.add_callback(callback);
}

/**
 * Add an event handler to the window.
 *
 * Usually an EventHandler is a user-movable camera. See EventHandler
 * documentation for more information.
 *
 * @param eventHandler A pointer to an existing instance of EventHandler.
 */
void Display::add_event_handler(const EventHandler::Ptr& eventHandler)
{
    if(eventHandler->uses_keyboard()) {
        this->add_key_callback(std::bind(&EventHandler::key_callback,
                                         eventHandler,
                                         std::placeholders::_1,
                                         std::placeholders::_2,
                                         std::placeholders::_3,
                                         std::placeholders::_4));
    }
    if(eventHandler->uses_mouse_position()) {
        this->add_mouse_position_callback(std::bind(&EventHandler::mouse_position_callback,
                                                    eventHandler,
                                                    std::placeholders::_1,
                                                    std::placeholders::_2));
    }
    if(eventHandler->uses_mouse_button()) {
        this->add_mouse_button_callback(std::bind(&EventHandler::mouse_button_callback,
                                                  eventHandler,
                                                  std::placeholders::_1,
                                                  std::placeholders::_2,
                                                  std::placeholders::_3));
    }
    if(eventHandler->uses_scroll()) {
        this->add_scroll_callback(std::bind(&EventHandler::scroll_callback,
                                            eventHandler,
                                            std::placeholders::_1,
                                            std::placeholders::_2));
    }
    eventHandler->set_window(window_);
}

}; //namespace display
}; //namespace rtac


