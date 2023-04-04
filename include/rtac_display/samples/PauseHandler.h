#ifndef _DEF_RTAC_DISPLAY_SAMPLES_PAUSE_HANDLER_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_PAUSE_HANDLER_H_

#include <rtac_display/EventHandler.h>

namespace rtac { namespace display { namespace samples {

class PauseHandler : public EventHandler
{
    public:

    using Ptr      = std::shared_ptr<PauseHandler>;
    using ConstPtr = std::shared_ptr<const PauseHandler>;

    protected:

    bool paused_;

    PauseHandler() : EventHandler(true), paused_(false) {}

    public:

    static Ptr Create() { return Ptr(new PauseHandler()); }

    bool is_paused() const { return paused_; }

    void key_callback(int key, int scancode, int action, int modes) {
        if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
            paused_ = !paused_;
            if(paused_) {
                std::cout << "rtac::display : paused" << std::endl;
            }
            else {
                std::cout << "rtac::display : resumed" << std::endl;
            }
        }
    }
};

} //namespace samples
} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_PAUSE_HANDLER_H_
