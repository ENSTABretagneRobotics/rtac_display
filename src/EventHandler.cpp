#include <rtac_display/EventHandler.h>

namespace rtac { namespace display {

/**
 * Constructor of EventHandler
 *
 * The parameters indicate whether the Display object should register a
 * callback of a specific type for this object.
 *
 * @param useKeyboard      indicates if this instance uses keyboard events.
 * @param useMousePosition indicates if this instance uses mouse position events.
 * @param useMouseButton   indicates if this instance uses mouse button events.
 * @param useScroll        indicates if this instance uses scroll events.
 */
EventHandler::EventHandler(bool useKeyboard,  bool useMousePosition,
                           bool useMouseButton, bool useScroll) :
    useKeyboard_(useKeyboard),
    useMousePosition_(useMousePosition),
    useMouseButton_(useMouseButton),
    useScroll_(useScroll)
{}

/**
 * Allocates a new instance of Event handler on the heap.
 *
 * The parameters indicate whether the Display object should register a
 * callback of a specific type for this object.
 *
 * @param useKeyboard      indicates if this instance uses keyboard events.
 * @param useMousePosition indicates if this instance uses mouse position events.
 * @param useMouseButton   indicates if this instance uses mouse button events.
 * @param useScroll        indicates if this instance uses scroll events.
 *
 * @return a shared pointer to the newly created Even handler instance.
 */
EventHandler::Ptr EventHandler::Create(bool useKeyboard,  bool useMousePosition,
                                       bool useMouseButton, bool useScroll)
{
    return Ptr(new EventHandler(useKeyboard, useMousePosition, useMouseButton, useScroll));
}

/**
 * @return true if this instance uses keyboard events.
 */
bool EventHandler::uses_keyboard() const
{
    return useKeyboard_;
}

/**
 * @return true if this instance uses mouse position events.
 */
bool EventHandler::uses_mouse_position() const
{
    return useMousePosition_;
}

/**
 * @return true if this instance uses mouse button events.
 */
bool EventHandler::uses_mouse_button() const
{
    return useMouseButton_;
}

/**
 * @return true if this instance uses scroll events.
 */
bool EventHandler::uses_scroll() const
{
    return useScroll_;
}

/**
 * Keyboard callback to be overriden in subclasses (does nothing by default).
 * 
 * @param key      keyboard key (value may depend on system configuration, i.e.
 *                 QWERTY, AZERTY...)
 * @param scancode platform-specific. Constant unique identifier for a key.
 *                 Constant regardless of system configuration, may/will vary
 *                 across platforms.
 * @param action   Either GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT.
 * @param mods     Modifier bits (CONTROL, SHIFT, ALT, SUPER ...)
 */
void EventHandler::key_callback(int key, int scancode, int action, int modes)
{}

/**
 * Mouse position callback to be overriden in subclasses (does nothing by
 * default).
 *
 * Mouse position is given relative to the lot-left corner of the window
 * content area. If platform supports it, sub-pixel cursor position is passed.
 *
 * @param x mouse position (left-right axis)
 * @param y mouse position (bottom-top axis)
 */
void EventHandler::mouse_position_callback(double x, double y)
{}

/**
 * Mouse button callback to be overriden in subclasses (does nothing by
 * default).
 * 
 * @param button Mouse button id, GLFW_MOUSE_BUTTON_(LEFT/RIGHT/MIDDLE/others)
 * @param action Either GLFW_PRESS, GLFW_RELEASE.
 * @param mods   Modifier bits (CONTROL, SHIFT, ALT, SUPER ...)
 */
void EventHandler::mouse_button_callback(int button, int action, int modes)
{}

/**
 * Scroll callback to be overriden in subclasses (does nothing by default).
 *
 * Mouse scroll is on y axis.
 *
 * @param x scroll offset
 * @param y scroll offset
 */
void EventHandler::scroll_callback(double x, double y)
{}

void EventHandler::set_window(const Window& window)
{
    window_ = window;
}

}; //namespace display
}; //namespace rtac

