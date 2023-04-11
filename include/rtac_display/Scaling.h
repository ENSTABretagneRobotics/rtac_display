#ifndef _DEF_RTAC_DISPLAY_SCALING_H_
#define _DEF_RTAC_DISPLAY_SCALING_H_

#include <rtac_base/types/Bounds.h>

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

    ScalingMode   mode()   const { return mode_;   }
    Bounds<float> range()  const { return range_;  }
    Bounds<float> limits() const { return limits_; }
    float         origin() const { return origin_; }

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

} //namespace display
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::display::ScalingMode& mode);
std::ostream& operator<<(std::ostream& os, const rtac::display::Scaling1D& scaling);

#endif //_DEF_RTAC_DISPLAY_SCALING_H_
