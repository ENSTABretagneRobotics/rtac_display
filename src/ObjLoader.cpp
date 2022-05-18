#include <rtac_display/ObjLoader.h>

namespace rtac { namespace display {

ObjLoader::ObjLoader(const std::string& datasetPath) :
    datasetPath_(datasetPath)
{
    std::cout << "Opening .obj dataset from :\n- " << datasetPath << std::endl;

    objPath_ = rtac::files::find_one(".*\\obj", datasetPath);
    if(objPath_ == rtac::files::NotFound) {
        std::ostringstream oss;
        oss << "Could not find .obj file in given dataset path " << datasetPath;
        throw std::runtime_error(oss.str());
    }
    std::cout << "Found .obj file :\n- " << objPath_ << std::endl;

    mtlPath_ = rtac::files::find_one(".*\\mtl", datasetPath);
    if(mtlPath_ == rtac::files::NotFound) {
        std::cout << "No .mtl file found. Ignoring." << std::endl;
    }
    else {
        std::cout << "Found .mtl file :\n- " << mtlPath_ << std::endl;
    }
}

void ObjLoader::load_geometry(unsigned int chunkSize)
{
    std::ifstream f(objPath_, std::ifstream::in);
    if(!f.is_open()) {
        std::ostringstream oss;
        oss << "Could not open file for reading : " << objPath_;
        throw std::runtime_error(oss.str());
    }
    
    ChunkContainer<Point>  points;
    ChunkContainer<UV>     uvs;
    ChunkContainer<Normal> normals;
    
    std::string line, token;
    while(std::getline(f, line)) {
        std::istringstream iss(line);
        std::getline(iss, token, ' ');
        //std::cout << "here " << token << std::endl << std::flush;
        if(token == "v") {
            Point p;
            iss >> p.x;
            iss >> p.y;
            iss >> p.z;
            points.push_back(p);
        }
        else if(token == "vt") {
            UV uv;
            iss >> uv.x;
            iss >> uv.y;
            uvs.push_back(uv);
        }
        else if(token == "vn") {
            Normal n;
            iss >> n.x;
            iss >> n.y;
            iss >> n.z;
            normals.push_back(n);
        }
    }

    points_  = points.to_vector();
    uvs_     = uvs.to_vector();
    normals_ = normals.to_vector();
}

}; //namespace display
}; //namespace rtac

