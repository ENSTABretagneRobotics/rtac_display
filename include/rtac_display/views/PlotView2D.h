#ifndef _DEF_RTAC_DISPLAY_VIEWS_PLOT_VIEW_2D_H_
#define _DEF_RTAC_DISPLAY_VIEWS_PLOT_VIEW_2D_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/Scaling.h>
#include <rtac_display/PlotData2D.h>
#include <rtac_display/views/View.h>

namespace rtac { namespace display {

class PlotView2D : public View
{
    public:

    using Ptr      = std::shared_ptr<PlotView2D>;
    using ConstPtr = std::shared_ptr<const PlotView2D>;

    using Mat4   = View::Mat4;
    using Shape  = View::Shape;

    protected:
    
    Scaling2D scaling_;
    bool      equalAspect_;

    PlotView2D(const Bounds<float>& xRange, const Bounds<float>& yRange);

    void update_projection() override;

    public:

    static Ptr Create(const Bounds<float>& xRange = Bounds<float>(0,1),
                      const Bounds<float>& yRange = Bounds<float>(0,1));
    static Ptr Create(const PlotData2D::ConstPtr& plotData);
                      

    const Scaling2D& scaling() const { return scaling_; }
          Scaling2D& scaling()       { return scaling_; }
    
    void enable_equal_aspect()  { equalAspect_ = true;  }
    void disable_equal_aspect() { equalAspect_ = false; }

    const Bounds<float>& x_range() const { return scaling_.x_range(); }
    const Bounds<float>& y_range() const { return scaling_.y_range(); }

    void update_ranges(const Bounds<float>& xRange, const Bounds<float>& yRange);
    void update_ranges(const PlotData2D::ConstPtr& plotData);
};

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_VIEWS_PLOT_VIEW_2D_H_
