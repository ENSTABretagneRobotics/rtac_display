#ifndef _DEF_RTAC_DISPLAY_EVENT_HANDLER_H_
#define _DEF_RTAC_DISPLAY_EVENT_HANDLER_H_

#include <iostream>
#include <iomanip>

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

class EventHandler
{
    public:

    using Ptr      = rtac::types::Handle<EventHandler>;
    using ConstPtr = rtac::types::Handle<const EventHandler>;
    using Window   = rtac::types::Handle<GLFWwindow>;

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
    
