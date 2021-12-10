#ifndef _DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_

#include <rtac_display/Display.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/samples/OrbitControls.h>

namespace rtac { namespace display { namespace samples {

class Display3D : public Display
{
    protected:

    PinholeView::Ptr   view_;
    OrbitControls::Ptr controls_;

    public:

    Display3D(int width = 800, int height = 600,
              const std::string& name = "rtac_display",
              const Display::Window& sharedContext = nullptr);
    Display3D(const Display::Window& sharedContext);

    PinholeView::Ptr view();
    OrbitControls::Ptr controls();
    PinholeView::ConstPtr view() const;
    OrbitControls::ConstPtr controls() const;
};

}; //namespace samples
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_
