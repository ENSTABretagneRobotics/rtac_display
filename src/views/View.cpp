#include <rtac_display/views/View.h>

namespace rtac { namespace display {

/**
 * Creates a new Instance of View on the heap and returns a smart pointer.
 *
 * @param a homogeneous 4x4 projection matrix (default is identity).
 *
 * @return a shared pointer to the newly created View instance.
 */
View::Ptr View::New(const Mat4& mat)
{
    return Ptr(new View(mat));
}

/**
 * Creates a new Instance of View on the heap and returns a smart pointer.
 *
 * @param a homogeneous 4x4 projection matrix (default is identity).
 *
 * @return a shared pointer to the newly created View instance.
 */
View::Ptr View::Create(const Mat4& mat)
{
    return Ptr(new View(mat));
}

/**
 * Constructor of View.
 *
 * By default the screen aspect ratio is 1 (screenSize is set to {1,1})
 *
 * @param a homogeneous 4x4 projection matrix (default is identity).
 */
View::View(const Mat4& mat) :
    screenSize_({1,1}),
    projectionMatrix_(mat)
{}

/** 
 * Virtual method to be overridden in sub-classes. The purpose of this method
 * is to compensate for the aspect ratio variations of the display area when
 * the window is resized. Aspect ratio is often encoded into the projection_
 * matrix.
 *
 * This method is called by View::set_screen_size **after** the attribute
 * screenSize_ was set.
 *
 * A typical way to compensate for the changing aspect ratio is to recalculate
 * the current aspect ratio from the projection matrix and then modify the
 * projection matrix with the new aspect ratio given by screenSize_.
 *
 * If this method is not overridden in sub-classes, the aspect ratio of the
 * render will change with the aspect ratio of the window (stretch vertically
 * or horizontally depending on the window size).
 */
void View::update_projection()
{}

/**
 * Called by a Display object when the display surface size changes.
 *
 * @param screen a rtac::types::Shape object containing the width and height
 *               of the display area.
 */
void View::set_screen_size(const Shape& screen)
{
    screenSize_ = screen;
    this->update_projection();
}

/**
 * @return the projection matrix.
 */
View::Mat4 View::projection_matrix() const
{
    return projectionMatrix_;
}

/**
 * @return the view matrix (product of the projection matrix with the pov
 *         matrix (to be changed to simplify the interface).
 */
View::Mat4 View::view_matrix() const
{
    return projectionMatrix_;
}

/**
 * @return the current screen size.
 */
View::Shape View::screen_size() const
{
    return screenSize_;
}

}; //namespace display
}; //namespace rtac

