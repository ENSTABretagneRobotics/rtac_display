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

    //mtlPath_ = rtac::files::find_one(".*\\mtl", datasetPath);
    //if(mtlPath_ == rtac::files::NotFound) {
    //    std::cout << "No .mtl file found. Ignoring." << std::endl;
    //}
    //else {
    //    std::cout << "Found .mtl file :\n- " << mtlPath_ << std::endl;
    //}
}

std::array<VertexId, 3> parse_face(const std::string& line)
{
    unsigned int slashCount = 0;
    for(char c : line) {
        if(c == '/') slashCount++;
    }
    
    std::size_t pos;
    std::string token;
    std::array<VertexId, 3> v = {0,0,0};
    switch(slashCount) {
        default:
            std::runtime_error("Invalid face format string");
            break;
        case 0:
            v[0].p = std::stoul(line, &pos) - 1;
            token = line.substr(pos);
            v[1].p = std::stoul(token, &pos) - 1;
            token = token.substr(pos);
            v[2].p = std::stoul(token, &pos) - 1;
            break;
        case 3:
            v[0].p = std::stoul(line, &pos) - 1;
            token = line.substr(pos + 1);
            v[0].u = std::stoul(token, &pos) - 1;
            token = token.substr(pos + 1);

            v[1].p = std::stoul(token, &pos) - 1;
            token = token.substr(pos + 1);
            v[1].u = std::stoul(token, &pos) - 1;
            token = token.substr(pos + 1);

            v[2].p = std::stoul(token, &pos) - 1;
            token = token.substr(pos + 1);
            v[2].u = std::stoul(token, &pos) - 1;
            break;
        case 6:
            v[0].p = std::stoul(line, &pos) - 1;
            if(line[pos + 1] == '/') {
                v[0].u = 0;
                token = line.substr(pos + 2);
            }
            else {
                token = line.substr(pos + 1);
                v[0].u = std::stoul(token, &pos) - 1;
                token = token.substr(pos + 1);
            }
            v[0].n  = std::stoul(token, &pos) - 1;
            token = token.substr(pos);

            v[1].p = std::stoul(token, &pos) - 1;
            if(token[pos + 1] == '/') {
                v[1].u = 0;
                token = token.substr(pos + 2);
            }
            else {
                token = token.substr(pos + 1);
                v[1].u = std::stoul(token, &pos) - 1;
                token = token.substr(pos + 1);
            }
            v[1].n  = std::stoul(token, &pos) - 1;
            token = token.substr(pos);

            v[2].p = std::stoul(token, &pos) - 1;
            if(token[pos + 1] == '/') {
                v[2].u = 0;
                token = token.substr(pos + 2);
            }
            else {
                token = token.substr(pos + 1);
                v[2].u = std::stoul(token, &pos) - 1;
                token = token.substr(pos + 1);
            }
            v[2].n  = std::stoul(token, &pos) - 1;
            break;
    }

    return v;
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
    ChunkContainer<Face>   faces;

    std::string currentMaterial = "";
    
    std::string line, token;
    unsigned int numFaces = 0;
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
        else if(token == "f") {
            std::getline(iss, token);
            auto v = parse_face(token);

            Face f;
            
            auto it0 = vertices_.insert(v[0]);
            if(it0.second) it0.first->id = vertices_.size() - 1;
            auto it1 = vertices_.insert(v[1]);
            if(it1.second) it1.first->id = vertices_.size() - 1;
            auto it2 = vertices_.insert(v[2]);
            if(it2.second) it2.first->id = vertices_.size() - 1;

            f.x = it0.first->id;
            f.y = it1.first->id;
            f.z = it2.first->id;
            
            faces.push_back(f);
        }
        else if(token == "usemtl") {
            std::getline(iss, token);
            if(currentMaterial.size() != 0) {
                faceGroups_[currentMaterial] = faces.to_vector();
                groupNames_.push_back(currentMaterial);
            }
            faces.clear();
            currentMaterial = token;
        }
        else if(token == "mtllib") {
            std::getline(iss, mtlPath_);
        }
    } // end of file

    if(currentMaterial.size() == 0) {
        faceGroups_["null_material"] = faces.to_vector();
        groupNames_.push_back(currentMaterial);
    }
    else {
        faceGroups_[currentMaterial] = faces.to_vector();
        groupNames_.push_back(currentMaterial);
    }
    points_  = points.to_vector();
    uvs_     = uvs.to_vector();
    normals_ = normals.to_vector();

    this->parse_mtl();
}

void ObjLoader::parse_mtl()
{
    if(mtlPath_ == "") return;

    auto path = rtac::files::find_one(std::string(".*") + mtlPath_, datasetPath_);
    if(path == rtac::files::NotFound) {
        std::cerr << "OBJ file " << objPath_ << " indicates a mtl file "
                  << mtlPath_ << " but none was found." << std::endl;
        mtlPath_ = "";
    }
    else {
        std::cout << "Found .mtl file :\n- " << mtlPath_ << std::endl;
        mtlPath_ = path;
    }

    std::ifstream f(mtlPath_, std::ifstream::in);
    if(!f.is_open()) {
        std::ostringstream oss;
        oss << "Could not open file for reading : " << mtlPath_;
        throw std::runtime_error(oss.str());
    }

    std::string line, token;
    MtlMaterial currentMtl;
    currentMtl.clear();
    std::size_t pos;
    while(std::getline(f, line)) {
        std::istringstream iss(line);
        std::getline(iss, token, ' ');

        if(token == "newmtl") {
            if(currentMtl.name != "") {
                materials_[currentMtl.name] = currentMtl;
                currentMtl.clear();
            }
            std::getline(iss, currentMtl.name);
        }
        else if(token == "Ka") {
            std::getline(iss, token);
            currentMtl.Ka.x = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Ka.y = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Ka.z = std::stof(token, &pos);
        }
        else if(token == "Kd") {
            std::getline(iss, token);
            currentMtl.Kd.x = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Kd.y = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Kd.z = std::stof(token, &pos);
        }
        else if(token == "Ks") {
            std::getline(iss, token);
            currentMtl.Ks.x = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Ks.y = std::stof(token, &pos);
            token = token.substr(pos);
            currentMtl.Ks.z = std::stof(token, &pos);
        }
        else if(token == "Ns") {
            std::getline(iss, token);
            currentMtl.Ns = std::stof(token);
        }
        else if(token == "d") {
            std::getline(iss, token);
            currentMtl.d = std::stof(token);
        }
        else if(token == "Tr") {
            std::getline(iss, token);
            currentMtl.d = 1.0f - std::stof(token);
        }
        else if(token == "illum") {
            std::getline(iss, token);
            currentMtl.illum = std::stoul(token);
        }
        else if(token == "map_Kd") {
            std::getline(iss, token);
            currentMtl.map_Kd = rtac::files::find_one(".*" + token, datasetPath_);
        }
    }
    if(currentMtl.name != "") {
        materials_[currentMtl.name] = currentMtl;
        currentMtl.clear();
    }

    // for(const auto& mat : materials_) {
    //     std::cout << mat.second << std::endl;
    // }
}

std::map<std::string,GLMesh::Ptr> ObjLoader::create_meshes()
{
    std::map<std::string,GLMesh::Ptr> meshes;

    for(auto name : groupNames_) {

        auto mesh = GLMesh::Create();

        mesh->points().resize(vertices_.size());
        {
            auto ptr = mesh->points().map();
            for(auto v : vertices_) {
                ptr[v.id] = points_[v.p];
            }
        }

        if(uvs_.size() > 0) {
            mesh->uvs().resize(vertices_.size());
            auto ptr = mesh->uvs().map();
            for(auto v : vertices_) {
                ptr[v.id] = uvs_[v.u];
            }
        }

        if(normals_.size() > 0) {
            mesh->normals().resize(vertices_.size());
            auto ptr = mesh->normals().map();
            for(auto v : vertices_) {
                ptr[v.id] = normals_[v.n];
            }
        }

        mesh->faces() = faceGroups_[name];
        meshes[name] = mesh;
    }
    return meshes;
}

}; //namespace display
}; //namespace rtac

