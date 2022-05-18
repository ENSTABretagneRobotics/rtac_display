#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_display/ObjLoader.h>
using namespace rtac::display;

int main()
{
    auto path = files::find_one(".*models3d/pyramide2_test01");
    ObjLoader parser(path);

    parser.load_geometry();

    cout << "points  : " << parser.points().size() << endl;
    cout << "uvs     : " << parser.uvs().size() << endl;
    cout << "normals : " << parser.normals().size() << endl;

    return 0;
}
