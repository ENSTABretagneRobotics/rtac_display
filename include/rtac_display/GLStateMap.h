#ifndef _DEF_RTAC_DISPLAY_GL_STATE_MAP_H_
#define _DEF_RTAC_DISPLAY_GL_STATE_MAP_H_

#include <cstring>
#include <unordered_map>

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLState.h>

namespace rtac { namespace display {

// struct UnknownCapability : public std::exception
// {
//     std::string message_;
// 
//     UnknownCapability(GLenum cap) {
//         std::ostringstream oss;
//         oss << "Unknown OpenGL capability : " << cap;
//         message_ = oss.str();
//     }
// 
//     const char* what() const {
//         return message_.cstr();
//     }
// }

class GLStateMap
{
    public:

    using Ptr      = rtac::types::Handle<GLState>;
    using ConstPtr = rtac::types::Handle<const GLState>;
    using StateMap = std::unordered_map<GLenum, bool>;

    //protected:

    StateMap stateMap_;

    bool& capability(GLenum cap)
    {
        auto it = stateMap_.find(cap);
        if(it == stateMap_.end())
            throw UnknownCapability(cap);
        else
            return it->second;
    }

    public:

    GLStateMap();

    static Ptr New() { return Ptr(new GLState()); }

    void enable(GLenum cap) {
        bool& state = this->capability(cap);
        if(!state) {
            glEnable(cap);
            state = true;
        }
    }

    void disable(GLenum cap) {
        bool& state = this->capability(cap);
        if(state) {
            glDisable(cap);
            state = false;
        }
    }

    bool is_enabled(GLenum cap) const
    {
        auto it = stateMap_.find(cap);
        if(it == stateMap_.end())
            throw UnknownCapability(cap);
        else
            return it->second;
    }
};

GLStateMap::GLStateMap()
{
    stateMap_[GL_ALPHA_TEST] = false;
    stateMap_[GL_AUTO_NORMAL] = false;
    stateMap_[GL_BLEND] = false;
    stateMap_[GL_CLIP_PLANE0] = false;
    stateMap_[GL_CLIP_PLANE0 + 1] = false;
    stateMap_[GL_CLIP_PLANE0 + 2] = false;
    stateMap_[GL_CLIP_PLANE0 + 3] = false;
    stateMap_[GL_CLIP_PLANE0 + 4] = false;
    stateMap_[GL_CLIP_PLANE0 + 5] = false;
    stateMap_[GL_CLIP_PLANE0 + 6] = false;
    stateMap_[GL_CLIP_PLANE0 + 7] = false;
    stateMap_[GL_COLOR_LOGIC_OP] = false;
    stateMap_[GL_COLOR_MATERIAL] = false;
    stateMap_[GL_COLOR_SUM] = false;
    stateMap_[GL_COLOR_TABLE] = false;
    stateMap_[GL_CONVOLUTION_1D] = false;
    stateMap_[GL_CONVOLUTION_2D] = false;
    stateMap_[GL_CULL_FACE] = false;
    stateMap_[GL_DEPTH_TEST] = false;
    stateMap_[GL_DITHER] = true;
    stateMap_[GL_FOG] = false;
    stateMap_[GL_HISTOGRAM] = false;
    stateMap_[GL_INDEX_LOGIC_OP] = false;
    stateMap_[GL_LIGHT0] = false;
    stateMap_[GL_LIGHT0 + 1] = false;
    stateMap_[GL_LIGHT0 + 2] = false;
    stateMap_[GL_LIGHT0 + 3] = false;
    stateMap_[GL_LIGHT0 + 4] = false;
    stateMap_[GL_LIGHT0 + 5] = false;
    stateMap_[GL_LIGHT0 + 6] = false;
    stateMap_[GL_LIGHT0 + 7] = false;
    stateMap_[GL_LIGHT0 + 8] = false;
    stateMap_[GL_LIGHT0 + 9] = false;
    stateMap_[GL_LIGHT0 + 10] = false;
    stateMap_[GL_LIGHT0 + 11] = false;
    stateMap_[GL_LIGHT0 + 12] = false;
    stateMap_[GL_LIGHT0 + 13] = false;
    stateMap_[GL_LIGHT0 + 14] = false;
    stateMap_[GL_LIGHT0 + 15] = false;
    stateMap_[GL_LIGHTING] = false;
    stateMap_[GL_LINE_SMOOTH] = false;
    stateMap_[GL_LINE_STIPPLE] = false;
    stateMap_[GL_MAP1_COLOR_4] = false;
    stateMap_[GL_MAP1_INDEX] = false;
    stateMap_[GL_MAP1_NORMAL] = false;
    stateMap_[GL_MAP1_TEXTURE_COORD_1] = false;
    stateMap_[GL_MAP1_TEXTURE_COORD_2] = false;
    stateMap_[GL_MAP1_TEXTURE_COORD_3] = false;
    stateMap_[GL_MAP1_TEXTURE_COORD_4] = false;
    stateMap_[GL_MAP1_VERTEX_3] = false;
    stateMap_[GL_MAP1_VERTEX_4] = false;
    stateMap_[GL_MAP2_COLOR_4] = false;
    stateMap_[GL_MAP2_INDEX] = false;
    stateMap_[GL_MAP2_NORMAL] = false;
    stateMap_[GL_MAP2_TEXTURE_COORD_1] = false;
    stateMap_[GL_MAP2_TEXTURE_COORD_2] = false;
    stateMap_[GL_MAP2_TEXTURE_COORD_3] = false;
    stateMap_[GL_MAP2_TEXTURE_COORD_4] = false;
    stateMap_[GL_MAP2_VERTEX_3] = false;
    stateMap_[GL_MAP2_VERTEX_4] = false;
    stateMap_[GL_MINMAX] = false;
    stateMap_[GL_MULTISAMPLE] = true;
    stateMap_[GL_NORMALIZE] = false;
    stateMap_[GL_POINT_SMOOTH] = false;
    stateMap_[GL_POINT_SPRITE] = false;
    stateMap_[GL_POLYGON_OFFSET_FILL] = false;
    stateMap_[GL_POLYGON_OFFSET_LINE] = false;
    stateMap_[GL_POLYGON_OFFSET_POINT] = false;
    stateMap_[GL_POLYGON_SMOOTH] = false;
    stateMap_[GL_POLYGON_STIPPLE] = false;
    stateMap_[GL_POST_COLOR_MATRIX_COLOR_TABLE] = false;
    stateMap_[GL_POST_CONVOLUTION_COLOR_TABLE] = false;
    stateMap_[GL_RESCALE_NORMAL] = false;
    stateMap_[GL_SAMPLE_ALPHA_TO_COVERAGE] = false;
    stateMap_[GL_SAMPLE_ALPHA_TO_ONE] = false;
    stateMap_[GL_SAMPLE_COVERAGE] = false;
    stateMap_[GL_SEPARABLE_2D] = false;
    stateMap_[GL_SCISSOR_TEST] = false;
    stateMap_[GL_STENCIL_TEST] = false;
    stateMap_[GL_TEXTURE_1D] = false;
    stateMap_[GL_TEXTURE_2D] = false;
    stateMap_[GL_TEXTURE_3D] = false;
    stateMap_[GL_TEXTURE_CUBE_MAP] = false;
    stateMap_[GL_TEXTURE_GEN_Q] = false;
    stateMap_[GL_TEXTURE_GEN_R] = false;
    stateMap_[GL_TEXTURE_GEN_S] = false;
    stateMap_[GL_TEXTURE_GEN_T] = false;
    stateMap_[GL_VERTEX_PROGRAM_POINT_SIZE] = false;
    stateMap_[GL_VERTEX_PROGRAM_TWO_SIDE] = false;
}

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_GL_STATE_H_

