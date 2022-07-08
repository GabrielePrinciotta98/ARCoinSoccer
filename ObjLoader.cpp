#include <fstream>
#include <sstream>
#include <opencv/cv.hpp>

#include "ObjLoader.h"
#include "lodepng.h"
#include <iostream>
#include <map>

void Texture::init()
{
    std::vector<unsigned char> image;
    if (valid())
    {

        unsigned error = lodepng::decode(image, width, height, file);
        if (error)
        {
            std::cout << "PNG decoder error (" + file + "): " << lodepng_error_text(error) << "\n";
            file = "";
        }

    }
    if (valid())
    {
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8_SNORM, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    initialized = true;
}

void Texture::bind()
{
    if (!initialized)
        std::cout << "Texture not initialized: " << file << "\n";

    if (valid())
        glBindTexture(GL_TEXTURE_2D, texture);
    else
        glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::drawQuad(float x, float y, float height)
{
    drawQuad(x, y, height * getWidth() / getHeight(), height);
}

void Texture::drawQuad(float x, float y, float width, float height)
{
    glEnable(GL_TEXTURE_2D);
    bind();
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(x - width, y - height);
    glTexCoord2f(1, 1);
    glVertex2f(x + width, y - height);
    glTexCoord2f(1, 0);
    glVertex2f(x + width, y + height);
    glTexCoord2f(0, 0);
    glVertex2f(x - width, y + height);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

bool Texture::valid()
{
    return !file.empty();
}

void ObjModel::init()
{
    texture.init();

    bool error = true;
    std::vector<cv::Vec3f> verts;
    std::vector<cv::Vec2f> texCos;
    std::vector<cv::Vec3f> norms;
    std::vector<cv::Vec4i> trisIndices;
    std::vector<cv::Vec4i> quadIndices;

    int currentColor = -1;
    std::map<std::string, int> matColors;

    bool quads;

	std::ifstream infile(objFile);
    if (infile.fail())
    {
        std::cout << "Unable to load model: " << objFile << "\n";
        error = true;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        
        std::istringstream iss(line);
        if (line.empty())
            continue;

        std::string start;
        iss >> start;

        if (start == "#" || start == "mtllib" || start == "g" || start == "s" || start == "o")
            continue;
        else if (start == "v")
        {
            float f1, f2, f3;
            iss >> f1 >> f2 >> f3;
            verts.push_back(cv::Vec3f(f1, f2, f3));
        }
        else if (start == "vn")
        {
            float f1, f2, f3;
            iss >> f1 >> f2 >> f3;
            norms.push_back(cv::Vec3f(f1, f2, f3));
        }
        else if (start == "vt")
        {
            float f1, f2;
            iss >> f1 >> f2;
            texCos.push_back(cv::Vec2f(f1, 1 - f2));
        }
        else if (start == "f")
        {
            std::string indexList, index;
           
            int count = std::count(line.begin(), line.end(), ' ');
            if (line[line.length() - 1] == ' ')
                count--;

            if (count != 3 && count != 4)
            {
                std::cout << "Face has too many vertices .obj (" << objFile << "): " << line << "\n";
                continue;
            }
            for (int i = 0; i < count; ++i)
            {
                iss >> indexList;
                std::stringstream ss(indexList);

                int i1 = 0, i2 = 0, i3 = 0;
                int pos = 0;
                if (std::getline(ss, index, '/'))
                    i1 = index.empty() ? 0 : std::stoi(index);
                if (std::getline(ss, index, '/'))
                    i2 = index.empty() ? 0 : std::stoi(index);
                if (std::getline(ss, index, '/'))
                    i3 = index.empty() ? 0 : std::stoi(index);

                if (i1 <= 0 || i1 > verts.size() || i2 > texCos.size() || i3 > norms.size())
                {
                    //std::cout << "Index out of bounds in .obj (" << objFile << "): " << line << "\n";
                    continue;
                }

                (count == 3 ? trisIndices : quadIndices).push_back(cv::Vec4i(i1 - 1, i2 - 1, i3 - 1, currentColor));
            }
        }
        else if (start == "usemtl")
        {
            std::string mat;
            iss >> mat;
            if (matColors.find(mat) == matColors.end())
            {
                std::cout << "Material \"" << mat  << "\" (" << objFile << ")\n";

                matColors[mat] = matColors.size() < colors.size() ? matColors.size() : -1;

            }

            currentColor = matColors[mat];
        }
        else
            std::cout << "Unable to parse line in .obj (" << objFile << "): " << line;
    }


    callList = glGenLists(1);
    glNewList(callList, GL_COMPILE);
    if (texture.valid())
    {
        glEnable(GL_TEXTURE_2D);
        texture.bind();
    }
    if (trisIndices.size() > 0)
    {
        glBegin(GL_TRIANGLES);
        for (const cv::Vec4i& i : trisIndices)
        {
            const cv::Vec3f& v = verts[i[0]];
            const cv::Vec2f& t = i[1] >= 0 ? texCos[i[1]] : cv::Vec2f(0, 0);
            const cv::Vec3f& n = i[2] >= 0 ? norms[i[2]] : cv::Vec3f(0, 0, 0);
            const cv::Vec4f& c = i[3] >= 0 ? colors[i[3]] : cv::Vec4f(1, 1, 1, 1);

            glColor4f(c[0], c[1], c[2], c[3]);
            glTexCoord2f(t[0], t[1]);
            glNormal3f(n[0], n[1], n[2]);
            glVertex3f(v[0], v[1], v[2]);
        }
        glEnd();
    }
    if (quadIndices.size() > 0)
    {
        glBegin(GL_QUADS);
        for (const cv::Vec4i& i : quadIndices)
        {
            const cv::Vec3f& v = verts[i[0]];
            const cv::Vec2f& t = i[1] >= 0 ? texCos[i[1]] : cv::Vec2f(0, 0);
            const cv::Vec3f& n = i[2] >= 0 ? norms[i[2]] : cv::Vec3f(0, 0, 0);
            const cv::Vec4f& c = i[3] >= 0 ? colors[i[3]] : cv::Vec4f(1, 1, 1, 1);

            glColor4f(c[0], c[1], c[2], c[3]);
            glTexCoord2f(t[0], t[1]);
            glNormal3f(n[0], n[1], n[2]);
            glVertex3f(v[0], v[1], v[2]);
        }
        glEnd();
    }
    glDisable(GL_TEXTURE_2D);
    glEndList();

    if (!error)
        std::cout << "Loaded model: " << objFile << "\n";

    initialized = true;
}

void ObjModel::draw()
{
    if (!initialized)
        std::cout << "Model not initialized: " << objFile << "\n";
    glCallList(callList);
}