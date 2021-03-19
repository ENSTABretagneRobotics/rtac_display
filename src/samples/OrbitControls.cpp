#include <rtac_display/samples/OrbitControls.h>

namespace rtac { namespace display { namespace samples {

using namespace rtac::types::indexing;

OrbitControls::OrbitControls(const View3D::Ptr& view,
                             const Vec3& target,
                             const Vec3& up) :
    EventHandler(false, true, true, false),
    view_(view),
    viewFrame_(Mat3::Identity()),
    locked_(false),
    target_(target),
    alpha_(0.005)
{
    viewFrame_(all,2) = up.normalized();
    viewFrame_(all,0) = geometry::find_orthogonal(up).normalized();
    viewFrame_(all,1) = up.cross(viewFrame_(all,0));
}

OrbitControls::Ptr OrbitControls::Create(const View3D::Ptr& view,
                                         const Vec3& target,
                                         const Vec3& up)
{
    return Ptr(new OrbitControls(view, target, up));
}

void OrbitControls::lock_view()
{
    glfwSetInputMode(window_.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    auto p = view_->pose().translation() - target_;

    rho_   = p.norm();
    phi_   = std::asin(p.normalized().dot(viewFrame_(all,2)));
    theta_ = std::atan2(p.dot(viewFrame_(all,1)), p.dot(viewFrame_(all,0)));
    
    locked_ = true;
}

void OrbitControls::unlock_view()
{
    glfwSetInputMode(window_.get(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    locked_ = false;
}

void OrbitControls::mouse_position_callback(double x, double y)
{
    if(!window_) {
        std::cerr << "Warning : window is null but callback was called." << std::endl;
        return;
    }

    if(!locked_) return;

    MousePosition dmouse = alpha_*(MousePosition({x,y}) - lastMouse_);
    lastMouse_ = MousePosition({x,y});

    theta_ -= dmouse[0];
    phi_   += dmouse[1];
    phi_ = std::max(std::min(phi_, 1.5f), -1.5f);

    Vec3 p = rho_*viewFrame_*Vec3({std::cos(theta_)*std::cos(phi_),
                                   std::sin(theta_)*std::cos(phi_),
                                   std::sin(phi_)}) + target_;

    view_->look_at(target_, p, viewFrame_(all,2));
}

void OrbitControls::mouse_button_callback(int button, int action, int modes)
{
    if(!window_) {
        std::cerr << "Warning : window is null but callback was called." << std::endl;
        return;
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT) {
        if(action == GLFW_PRESS) {
            this->lock_view();
            glfwGetCursorPos(window_.get(), &lastMouse_[0], &lastMouse_[1]);
        }
        else {
            this->unlock_view();
        }
    }
}

}; //namespace samples
}; //namespace display
}; //namespace rtac
