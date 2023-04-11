#ifndef _DEF_RTAC_DISPLAY_SCALING_H_
#define _DEF_RTAC_DISPLAY_SCALING_H_

#include <iostream>

#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Point.h>

namespace rtac { namespace display {

struct ScalingMode
{
    bool autoscale_; // will update range on update
    bool memory_;    // New range contains old range
    bool useOrigin_; // keep origin in range
    bool useLimits_; // use absolute limits when autoscaling

    ScalingMode(bool autoscale = true,
                bool memory    = false,
                bool useOrigin = false,
                bool useLimits = false);
};

class Scaling1D
{
    protected:
    
    ScalingMode   mode_;
    Bounds<float> range_;
    Bounds<float> limits_;
    float         origin_;

    public:

    Scaling1D(const ScalingMode& mode = ScalingMode(),
              const Bounds<float>& range = Bounds<float>(0.0f,1.0f));
    Scaling1D(const Bounds<float>& range);

    const ScalingMode&   mode()   const { return mode_;   }
    const Bounds<float>& range()  const { return range_;  }
    const Bounds<float>& limits() const { return limits_; }
    float origin() const { return origin_; }

    void enable_autoscale()  { mode_.autoscale_ = true;  }
    void disable_autoscale() { mode_.autoscale_ = false; }
    void enable_memory()     { mode_.memory_ = true;  }
    void disable_memory()    { mode_.memory_ = false; }

    void enable_origin();
    void enable_origin(float origin);
    void set_origin(float origin, bool enableOrigin = true);
    void disable_origin() { mode_.useOrigin_ = false; }

    void enable_limits();
    void enable_limits(const Bounds<float>& limits);
    void set_limits(const Bounds<float>& limits, bool enableLimits = true);
    void disable_limits() { mode_.useLimits_ = false; }

    void set_range(const Bounds<float>& range, bool disableAutoscale = true);

    bool update(Bounds<float> range);
};

class Scaling2D
{
    protected:

    Scaling1D xScaling_;
    Scaling1D yScaling_;

    public:

    Scaling2D(const ScalingMode& mode = ScalingMode(),
              const Bounds<float>& xRange = Bounds<float>(0.0f, 1.0f),
              const Bounds<float>& yRange = Bounds<float>(0.0f, 1.0f));
    Scaling2D(const Bounds<float>& xRange, const Bounds<float>& yRange);

    const Scaling1D& x_scaling() const { return xScaling_; }
          Scaling1D& x_scaling()       { return xScaling_; }
    const Scaling1D& y_scaling() const { return yScaling_; }
          Scaling1D& y_scaling()       { return yScaling_; }

    const Bounds<float>& x_range()  const { return xScaling_.range();  }
    const Bounds<float>& y_range()  const { return yScaling_.range();  }
    const Bounds<float>& x_limits() const { return xScaling_.limits(); }
    const Bounds<float>& y_limits() const { return yScaling_.limits(); }
    Point2<float> origin()   const;

    void enable_autoscale();
    void disable_autoscale();
    void enable_memory();
    void disable_memory();

    void enable_origin();
    void enable_origin(Point2<float> origin);
    void set_origin(Point2<float> origin, bool enableOrigin = true);
    void disable_origin();

    void enable_limits();
    void enable_limits(const Bounds<float>& xLimits, const Bounds<float>& yLimits);
    void set_limits(const Bounds<float>& xLimits, const Bounds<float>& yLimits,
                    bool enableLimits = true);
    void disable_limits();

    void set_range(const Bounds<float>& xRange, const Bounds<float>& yRange,
                   bool disableAutoscale = true);

    bool update(const Bounds<float>& xRange, const Bounds<float>& yRange);
};

} //namespace display
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::display::ScalingMode& mode);
std::ostream& operator<<(std::ostream& os, const rtac::display::Scaling1D& scaling);
std::ostream& operator<<(std::ostream& os, const rtac::display::Scaling2D& scaling);

#endif //_DEF_RTAC_DISPLAY_SCALING_H_
