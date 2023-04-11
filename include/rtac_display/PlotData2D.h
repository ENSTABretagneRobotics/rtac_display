#ifndef _DEF_RTAC_DISPLAY_PLOT_DATA_2D_H_
#define _DEF_RTAC_DISPLAY_PLOT_DATA_2D_H_

#include <memory>

#include <rtac_base/types/Bounds.h>

#include <rtac_display/GLVector.h>
#include <rtac_display/GLReductor.h>

namespace rtac { namespace display {

class PlotData2D
{
    public:

    using Ptr      = std::shared_ptr<PlotData2D>;
    using ConstPtr = std::shared_ptr<const PlotData2D>;
    
    protected:

    GLVector<float> x_;
    GLVector<float> y_;
    Bounds<float>   xRange_;
    Bounds<float>   yRange_;

    GLReductor reductor_;

    PlotData2D();

    public:

    static Ptr Create() { return Ptr(new PlotData2D()); }
    template <template<typename>class VectorT>
    static Ptr Create(const VectorT<float>& x, const VectorT<float>& y);
    template <template<typename>class VectorT>
    static Ptr Create(const VectorT<float>& y);

    unsigned int size() const { return y_.size(); }
    const GLVector<float>& x() const { return x_; }
    const GLVector<float>& y() const { return y_; }
    const Bounds<float>&   x_range() const { return xRange_; }
    const Bounds<float>&   y_range() const { return yRange_; }

    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& x, const VectorT<float>& y);
    template <template<typename>class VectorT>
    bool set_data(const VectorT<float>& y);

    void update_ranges();
};

template <template<typename>class VectorT>
PlotData2D::Ptr PlotData2D::Create(const VectorT<float>& x, const VectorT<float>& y)
{
    auto res = Create();
    res->set_data(x,y);
    return res;
}

template <template<typename>class VectorT>
PlotData2D::Ptr PlotData2D::Create(const VectorT<float>& y)
{
    auto res = Create();
    res->set_data(y);
    return res;
}

template <template<typename>class VectorT>
bool PlotData2D::set_data(const VectorT<float>& x, const VectorT<float>& y)
{
    if(x.size() != y.size() || x.size() == 0) {
        return false;
    }
    x_ = x;
    y_ = y;
    this->update_ranges();
    return true;
}

template <template<typename>class VectorT>
bool PlotData2D::set_data(const VectorT<float>& y)
{
    if(y.size() == 0) {
        return false;
    }
    x_.clear();
    y_ = y;
    this->update_ranges();
    return true;
}

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_PLOT_DATA_2D_H_
