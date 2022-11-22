#ifndef _DEF_RTAC_DISPLAY_EVENT_HANDLER_H_
#define _DEF_RTAC_DISPLAY_EVENT_HANDLER_H_

#include <memory>
#include <iostream>
#include <iomanip>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

/**
 * Base class to handle event in Display object.
 *
 * When registered into a Display object, the callback methods of this object
 * will be registrered to receive window events. A custom event handler can be
 * created by subclassing EventHandler and overriding one or several of
 * callback methods. The callback methods which can be overridden are :
 * - key_callback            : called on keyboard event
 * - mouse_position_callback : called on mouse motion event
 * - mouse_button_callback   : called on mouse button event
 * - scroll_callback         : called on scroll event
 */
class EventHandler
{
    public:

    using Ptr      = std::shared_ptr<EventHandler>;
    using ConstPtr = std::shared_ptr<const EventHandler>;
    using Window   = std::shared_ptr<GLFWwindow>;

    protected:
    
    bool   useKeyboard_;
    bool   useMousePosition_;
    bool   useMouseButton_;
    bool   useScroll_;
    Window window_;

    public:

    EventHandler(bool useKeyboard = false,  bool useMousePosition = false,
                 bool useMouseButton = false, bool useScroll = false);

    static Ptr Create(bool useKeyboard = false,  bool useMousePosition = false,
                      bool useMouseButton = false, bool useScroll = false);

    bool uses_keyboard() const;
    bool uses_mouse_position() const;
    bool uses_mouse_button() const;
    bool uses_scroll() const;

    virtual void key_callback(int key, int scancode, int action, int modes);
    virtual void mouse_position_callback(double x, double y);
    virtual void mouse_button_callback(int button, int action, int modes);
    virtual void scroll_callback(double x, double y);

    void set_window(const Window& window);
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_EVENT_HANDLER_H_
    
