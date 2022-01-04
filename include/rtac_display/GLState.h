#ifndef _DEF_RTAC_DISPLAY_GL_STATE_H_
#define _DEF_RTAC_DISPLAY_GL_STATE_H_

#include <cstring>

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>

namespace rtac { namespace display {

struct UnknownCapability : public std::exception
{
    std::string message_;

    UnknownCapability(GLenum cap) {
        std::ostringstream oss;
        oss << "Unknown OpenGL capability : " << cap;
        message_ = oss.str();
    }

    const char* what() const throw() {
        return message_.c_str();
    }
};

class GLState
{
    public:

    using Ptr      = rtac::types::Handle<GLState>;
    using ConstPtr = rtac::types::Handle<const GLState>;

    GLState(); 

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

    bool is_enabled(GLenum cap) const;

    protected:

    bool& capability(GLenum cap);

    bool alphaTest_;
    bool autoNormal_;
    bool blend_;
    bool clipPlane0_;
    bool clipPlane1_;
    bool clipPlane2_;
    bool clipPlane3_;
    bool clipPlane4_;
    bool clipPlane5_;
    bool clipPlane6_;
    bool clipPlane7_;
    bool colorLogicOp_;
    bool colorMaterial_;
    bool colorSum_;
    bool colorTable_;
    bool convolution1d_;
    bool convolution2d_;
    bool cullFace_;
    bool depthTest_;
    bool dither_;
    bool fog_;
    bool histogram_;
    bool indexLogicOp_;
    bool light0_;
    bool light1_;
    bool light2_;
    bool light3_;
    bool light4_;
    bool light5_;
    bool light6_;
    bool light7_;
    bool light8_;
    bool light9_;
    bool light10_;
    bool light11_;
    bool light12_;
    bool light13_;
    bool light14_;
    bool light15_;
    bool lighting_;
    bool lineSmooth_;
    bool lineStipple_;
    bool map1Color4_;
    bool map1Index_;
    bool map1Normal_;
    bool map1TextureCoord1_;
    bool map1TextureCoord2_;
    bool map1TextureCoord3_;
    bool map1TextureCoord4_;
    bool map1Vertex3_;
    bool map1Vertex4_;
    bool map2Color4_;
    bool map2Index_;
    bool map2Normal_;
    bool map2TextureCoord1_;
    bool map2TextureCoord2_;
    bool map2TextureCoord3_;
    bool map2TextureCoord4_;
    bool map2Vertex3_;
    bool map2Vertex4_;
    bool minmax_;
    bool multisample_;
    bool normalize_;
    bool pointSmooth_;
    bool pointSprite_;
    bool polygonOffsetFill_;
    bool polygonOffsetLine_;
    bool polygonOffsetPoint_;
    bool polygonSmooth_;
    bool polygonStipple_;
    bool postColorMatrixColorTable_;
    bool postConvolutionColorTable_;
    bool rescaleNormal_;
    bool sampleAlphaToCoverage_;
    bool sampleAlphaToOne_;
    bool sampleCoverage_;
    bool separable2d_;
    bool scissorTest_;
    bool stencilTest_;
    bool texture1d_;
    bool texture2d_;
    bool texture3d_;
    bool textureCubeMap_;
    bool textureGenQ_;
    bool textureGenR_;
    bool textureGenS_;
    bool textureGenT_;
    bool vertexProgramPointSize_;
    bool vertexProgramTwoSide_;
};

inline GLState::GLState()
{
    std::memset(this, 0, sizeof(GLState));
    dither_      = true;
    multisample_ = true;
}

inline bool GLState::is_enabled(GLenum cap) const
{
    switch(cap) {
        default: throw UnknownCapability(cap); break;
        case GL_ALPHA_TEST:                    return alphaTest_;                 break;
        case GL_AUTO_NORMAL:                   return autoNormal_;                break;
        case GL_BLEND:                         return blend_;                     break;
        case GL_CLIP_PLANE0:                   return clipPlane0_;                break;
        case GL_CLIP_PLANE0 + 1:               return clipPlane1_;                break;
        case GL_CLIP_PLANE0 + 2:               return clipPlane2_;                break;
        case GL_CLIP_PLANE0 + 3:               return clipPlane3_;                break;
        case GL_CLIP_PLANE0 + 4:               return clipPlane4_;                break;
        case GL_CLIP_PLANE0 + 5:               return clipPlane5_;                break;
        case GL_CLIP_PLANE0 + 6:               return clipPlane6_;                break;
        case GL_CLIP_PLANE0 + 7:               return clipPlane7_;                break;
        case GL_COLOR_LOGIC_OP:                return colorLogicOp_;              break;
        case GL_COLOR_MATERIAL:                return colorMaterial_;             break;
        case GL_COLOR_SUM:                     return colorSum_;                  break;
        case GL_COLOR_TABLE:                   return colorTable_;                break;
        case GL_CONVOLUTION_1D:                return convolution1d_;             break;
        case GL_CONVOLUTION_2D:                return convolution2d_;             break;
        case GL_CULL_FACE:                     return cullFace_;                  break;
        case GL_DEPTH_TEST:                    return depthTest_;                 break;
        case GL_DITHER:                        return dither_;                    break;
        case GL_FOG:                           return fog_;                       break;
        case GL_HISTOGRAM:                     return histogram_;                 break;
        case GL_INDEX_LOGIC_OP:                return indexLogicOp_;              break;
        case GL_LIGHT0:                        return light0_;                    break;
        case GL_LIGHT0 + 1:                    return light1_;                    break;
        case GL_LIGHT0 + 2:                    return light2_;                    break;
        case GL_LIGHT0 + 3:                    return light3_;                    break;
        case GL_LIGHT0 + 4:                    return light4_;                    break;
        case GL_LIGHT0 + 5:                    return light5_;                    break;
        case GL_LIGHT0 + 6:                    return light6_;                    break;
        case GL_LIGHT0 + 7:                    return light7_;                    break;
        case GL_LIGHT0 + 8:                    return light8_;                    break;
        case GL_LIGHT0 + 9:                    return light9_;                    break;
        case GL_LIGHT0 + 10:                   return light10_;                   break;
        case GL_LIGHT0 + 11:                   return light11_;                   break;
        case GL_LIGHT0 + 12:                   return light12_;                   break;
        case GL_LIGHT0 + 13:                   return light13_;                   break;
        case GL_LIGHT0 + 14:                   return light14_;                   break;
        case GL_LIGHT0 + 15:                   return light15_;                   break;
        case GL_LIGHTING:                      return lighting_;                  break;
        case GL_LINE_SMOOTH:                   return lineSmooth_;                break;
        case GL_LINE_STIPPLE:                  return lineStipple_;               break;
        case GL_MAP1_COLOR_4:                  return map1Color4_;                break;
        case GL_MAP1_INDEX:                    return map1Index_;                 break;
        case GL_MAP1_NORMAL:                   return map1Normal_;                break;
        case GL_MAP1_TEXTURE_COORD_1:          return map1TextureCoord1_;         break;
        case GL_MAP1_TEXTURE_COORD_2:          return map1TextureCoord2_;         break;
        case GL_MAP1_TEXTURE_COORD_3:          return map1TextureCoord3_;         break;
        case GL_MAP1_TEXTURE_COORD_4:          return map1TextureCoord4_;         break;
        case GL_MAP1_VERTEX_3:                 return map1Vertex3_;               break;
        case GL_MAP1_VERTEX_4:                 return map1Vertex4_;               break;
        case GL_MAP2_COLOR_4:                  return map2Color4_;                break;
        case GL_MAP2_INDEX:                    return map2Index_;                 break;
        case GL_MAP2_NORMAL:                   return map2Normal_;                break;
        case GL_MAP2_TEXTURE_COORD_1:          return map2TextureCoord1_;         break;
        case GL_MAP2_TEXTURE_COORD_2:          return map2TextureCoord2_;         break;
        case GL_MAP2_TEXTURE_COORD_3:          return map2TextureCoord3_;         break;
        case GL_MAP2_TEXTURE_COORD_4:          return map2TextureCoord4_;         break;
        case GL_MAP2_VERTEX_3:                 return map2Vertex3_;               break;
        case GL_MAP2_VERTEX_4:                 return map2Vertex4_;               break;
        case GL_MINMAX:                        return minmax_;                    break;
        case GL_MULTISAMPLE:                   return multisample_;               break;
        case GL_NORMALIZE:                     return normalize_;                 break;
        case GL_POINT_SMOOTH:                  return pointSmooth_;               break;
        case GL_POINT_SPRITE:                  return pointSprite_;               break;
        case GL_POLYGON_OFFSET_FILL:           return polygonOffsetFill_;         break;
        case GL_POLYGON_OFFSET_LINE:           return polygonOffsetLine_;         break;
        case GL_POLYGON_OFFSET_POINT:          return polygonOffsetPoint_;        break;
        case GL_POLYGON_SMOOTH:                return polygonSmooth_;             break;
        case GL_POLYGON_STIPPLE:               return polygonStipple_;            break;
        case GL_POST_COLOR_MATRIX_COLOR_TABLE: return postColorMatrixColorTable_; break;
        case GL_POST_CONVOLUTION_COLOR_TABLE:  return postConvolutionColorTable_; break;
        case GL_RESCALE_NORMAL:                return rescaleNormal_;             break;
        case GL_SAMPLE_ALPHA_TO_COVERAGE:      return sampleAlphaToCoverage_;     break;
        case GL_SAMPLE_ALPHA_TO_ONE:           return sampleAlphaToOne_;          break;
        case GL_SAMPLE_COVERAGE:               return sampleCoverage_;            break;
        case GL_SEPARABLE_2D:                  return separable2d_;               break;
        case GL_SCISSOR_TEST:                  return scissorTest_;               break;
        case GL_STENCIL_TEST:                  return stencilTest_;               break;
        case GL_TEXTURE_1D:                    return texture1d_;                 break;
        case GL_TEXTURE_2D:                    return texture2d_;                 break;
        case GL_TEXTURE_3D:                    return texture3d_;                 break;
        case GL_TEXTURE_CUBE_MAP:              return textureCubeMap_;            break;
        case GL_TEXTURE_GEN_Q:                 return textureGenQ_;               break;
        case GL_TEXTURE_GEN_R:                 return textureGenR_;               break;
        case GL_TEXTURE_GEN_S:                 return textureGenS_;               break;
        case GL_TEXTURE_GEN_T:                 return textureGenT_;               break;
        case GL_VERTEX_PROGRAM_POINT_SIZE:     return vertexProgramPointSize_;    break;
        case GL_VERTEX_PROGRAM_TWO_SIDE:       return vertexProgramTwoSide_;      break;
    }
}

inline bool& GLState::capability(GLenum cap)
{
    switch(cap) {
        default: throw UnknownCapability(cap); break;
        case GL_ALPHA_TEST:                    return alphaTest_;                 break;
        case GL_AUTO_NORMAL:                   return autoNormal_;                break;
        case GL_BLEND:                         return blend_;                     break;
        case GL_CLIP_PLANE0:                   return clipPlane0_;                break;
        case GL_CLIP_PLANE0 + 1:               return clipPlane1_;                break;
        case GL_CLIP_PLANE0 + 2:               return clipPlane2_;                break;
        case GL_CLIP_PLANE0 + 3:               return clipPlane3_;                break;
        case GL_CLIP_PLANE0 + 4:               return clipPlane4_;                break;
        case GL_CLIP_PLANE0 + 5:               return clipPlane5_;                break;
        case GL_CLIP_PLANE0 + 6:               return clipPlane6_;                break;
        case GL_CLIP_PLANE0 + 7:               return clipPlane7_;                break;
        case GL_COLOR_LOGIC_OP:                return colorLogicOp_;              break;
        case GL_COLOR_MATERIAL:                return colorMaterial_;             break;
        case GL_COLOR_SUM:                     return colorSum_;                  break;
        case GL_COLOR_TABLE:                   return colorTable_;                break;
        case GL_CONVOLUTION_1D:                return convolution1d_;             break;
        case GL_CONVOLUTION_2D:                return convolution2d_;             break;
        case GL_CULL_FACE:                     return cullFace_;                  break;
        case GL_DEPTH_TEST:                    return depthTest_;                 break;
        case GL_DITHER:                        return dither_;                    break;
        case GL_FOG:                           return fog_;                       break;
        case GL_HISTOGRAM:                     return histogram_;                 break;
        case GL_INDEX_LOGIC_OP:                return indexLogicOp_;              break;
        case GL_LIGHT0:                        return light0_;                    break;
        case GL_LIGHT0 + 1:                    return light1_;                    break;
        case GL_LIGHT0 + 2:                    return light2_;                    break;
        case GL_LIGHT0 + 3:                    return light3_;                    break;
        case GL_LIGHT0 + 4:                    return light4_;                    break;
        case GL_LIGHT0 + 5:                    return light5_;                    break;
        case GL_LIGHT0 + 6:                    return light6_;                    break;
        case GL_LIGHT0 + 7:                    return light7_;                    break;
        case GL_LIGHT0 + 8:                    return light8_;                    break;
        case GL_LIGHT0 + 9:                    return light9_;                    break;
        case GL_LIGHT0 + 10:                   return light10_;                   break;
        case GL_LIGHT0 + 11:                   return light11_;                   break;
        case GL_LIGHT0 + 12:                   return light12_;                   break;
        case GL_LIGHT0 + 13:                   return light13_;                   break;
        case GL_LIGHT0 + 14:                   return light14_;                   break;
        case GL_LIGHT0 + 15:                   return light15_;                   break;
        case GL_LIGHTING:                      return lighting_;                  break;
        case GL_LINE_SMOOTH:                   return lineSmooth_;                break;
        case GL_LINE_STIPPLE:                  return lineStipple_;               break;
        case GL_MAP1_COLOR_4:                  return map1Color4_;                break;
        case GL_MAP1_INDEX:                    return map1Index_;                 break;
        case GL_MAP1_NORMAL:                   return map1Normal_;                break;
        case GL_MAP1_TEXTURE_COORD_1:          return map1TextureCoord1_;         break;
        case GL_MAP1_TEXTURE_COORD_2:          return map1TextureCoord2_;         break;
        case GL_MAP1_TEXTURE_COORD_3:          return map1TextureCoord3_;         break;
        case GL_MAP1_TEXTURE_COORD_4:          return map1TextureCoord4_;         break;
        case GL_MAP1_VERTEX_3:                 return map1Vertex3_;               break;
        case GL_MAP1_VERTEX_4:                 return map1Vertex4_;               break;
        case GL_MAP2_COLOR_4:                  return map2Color4_;                break;
        case GL_MAP2_INDEX:                    return map2Index_;                 break;
        case GL_MAP2_NORMAL:                   return map2Normal_;                break;
        case GL_MAP2_TEXTURE_COORD_1:          return map2TextureCoord1_;         break;
        case GL_MAP2_TEXTURE_COORD_2:          return map2TextureCoord2_;         break;
        case GL_MAP2_TEXTURE_COORD_3:          return map2TextureCoord3_;         break;
        case GL_MAP2_TEXTURE_COORD_4:          return map2TextureCoord4_;         break;
        case GL_MAP2_VERTEX_3:                 return map2Vertex3_;               break;
        case GL_MAP2_VERTEX_4:                 return map2Vertex4_;               break;
        case GL_MINMAX:                        return minmax_;                    break;
        case GL_MULTISAMPLE:                   return multisample_;               break;
        case GL_NORMALIZE:                     return normalize_;                 break;
        case GL_POINT_SMOOTH:                  return pointSmooth_;               break;
        case GL_POINT_SPRITE:                  return pointSprite_;               break;
        case GL_POLYGON_OFFSET_FILL:           return polygonOffsetFill_;         break;
        case GL_POLYGON_OFFSET_LINE:           return polygonOffsetLine_;         break;
        case GL_POLYGON_OFFSET_POINT:          return polygonOffsetPoint_;        break;
        case GL_POLYGON_SMOOTH:                return polygonSmooth_;             break;
        case GL_POLYGON_STIPPLE:               return polygonStipple_;            break;
        case GL_POST_COLOR_MATRIX_COLOR_TABLE: return postColorMatrixColorTable_; break;
        case GL_POST_CONVOLUTION_COLOR_TABLE:  return postConvolutionColorTable_; break;
        case GL_RESCALE_NORMAL:                return rescaleNormal_;             break;
        case GL_SAMPLE_ALPHA_TO_COVERAGE:      return sampleAlphaToCoverage_;     break;
        case GL_SAMPLE_ALPHA_TO_ONE:           return sampleAlphaToOne_;          break;
        case GL_SAMPLE_COVERAGE:               return sampleCoverage_;            break;
        case GL_SEPARABLE_2D:                  return separable2d_;               break;
        case GL_SCISSOR_TEST:                  return scissorTest_;               break;
        case GL_STENCIL_TEST:                  return stencilTest_;               break;
        case GL_TEXTURE_1D:                    return texture1d_;                 break;
        case GL_TEXTURE_2D:                    return texture2d_;                 break;
        case GL_TEXTURE_3D:                    return texture3d_;                 break;
        case GL_TEXTURE_CUBE_MAP:              return textureCubeMap_;            break;
        case GL_TEXTURE_GEN_Q:                 return textureGenQ_;               break;
        case GL_TEXTURE_GEN_R:                 return textureGenR_;               break;
        case GL_TEXTURE_GEN_S:                 return textureGenS_;               break;
        case GL_TEXTURE_GEN_T:                 return textureGenT_;               break;
        case GL_VERTEX_PROGRAM_POINT_SIZE:     return vertexProgramPointSize_;    break;
        case GL_VERTEX_PROGRAM_TWO_SIDE:       return vertexProgramTwoSide_;      break;
    }
}

}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_GL_STATE_H_
