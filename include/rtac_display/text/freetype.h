#ifndef _DEF_RTAC_DISPLAY_TEXT_FREETYPE_H_
#define _DEF_RTAC_DISPLAY_TEXT_FREETYPE_H_

#include <iostream>
#include <memory>

#define FT_CONFIG_OPTION_SUBPIXEL_RENDERING
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_LCD_FILTER_H

namespace rtac { namespace display { namespace text {

/**
 * Simple class to handle a FT_Library.
 */
class Library
{
    public:

    using Ptr      = std::shared_ptr<Library>;
    using ConstPtr = std::shared_ptr<const Library>;

    protected:

    FT_Library library_;

    Library();

    public:

    ~Library();

    static Ptr Create();

    operator FT_Library();
};

}; //namespace text
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_TEXT_FREETYPE_H_
