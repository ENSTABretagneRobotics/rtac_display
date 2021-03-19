#include <rtac_display/EventHandler.h>

namespace rtac { namespace display {

EventHandler::EventHandler(bool useKeyboard,  bool useMousePosition,
                           bool useMouseButton, bool useScroll) :
    useKeyboard_(useKeyboard),
    useMousePosition_(useMousePosition),
    useMouseButton_(useMouseButton),
    useScroll_(useScroll)
{}

EventHandler::Ptr EventHandler::Create(bool useKeyboard,  bool useMousePosition,
                                       bool useMouseButton, bool useScroll)
{
    return Ptr(new EventHandler(useKeyboard, useMousePosition, useMouseButton, useScroll));
}

bool EventHandler::uses_keyboard() const
{
    return useKeyboard_;
}

bool EventHandler::uses_mouse_position() const
{
    return useMousePosition_;
}

bool EventHandler::uses_mouse_button() const
{
    return useMouseButton_;
}

bool EventHandler::uses_scroll() const
{
    return useScroll_;
}

void EventHandler::key_callback(int key, int scancode, int action, int modes)
{}

void EventHandler::mouse_position_callback(double x, double y)
{}

void EventHandler::mouse_button_callback(int button, int action, int modes)
{}

void EventHandler::scroll_callback(double x, double y)
{}

void EventHandler::set_window(const Window& window)
{
    window_ = window;
}

}; //namespace display
}; //namespace rtac

