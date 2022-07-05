#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv/cv.hpp>

#include "DrawPrimitives.h"
#include "MarkerTracker.h"
#include "CoinTracker.h"
#include "PoseEstimation.h"


using namespace cv;
using namespace std;

//callback for resizing the window
void reshape(GLFWwindow* window, int width, int height);
//openGL initialization
void initGL(int argc, char* argv[]);
//rendering
void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Vec3f>& circles, bool marker);

const double fov =  63.75;
const double downscale = 0.25;

int main(int argc, char* argv[])
{
    //Set up OpenCV
    cv::VideoCapture cap("CoinSoccerMovie.mp4");
   //cv::VideoCapture cap(1);
    if (!cap.isOpened())
    {
        std::cout << "No capture" << std::endl;
        return -1;
    }

    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 8);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT), "Coin Soccer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, reshape);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    reshape(window, width, height);

    initGL(argc, argv);

    const double kMarkerSize = 0.045;
    MarkerTracker markerTracker(kMarkerSize, 120, 120);
    CoinTracker coinTracker(24 * downscale);
    std::vector<Vec3f> circles;

    cv::Mat img_background, img_background_resized, img_background2;
    float resultMatrix[16];
    while (!glfwWindowShouldClose(window))
    {
        if (!cap.read(img_background))
        {
            cap.set(CAP_PROP_POS_FRAMES, 0); // Loop back to frame 0
            if (!cap.read(img_background))
                break;
        }

        img_background2 = img_background.clone();
        cv::resize(img_background, img_background_resized, Size(), downscale, downscale, CV_INTER_LINEAR);
        coinTracker.findCoins(img_background_resized, circles);
        bool marker = markerTracker.findMarker(img_background, resultMatrix);
        display(window, img_background2, resultMatrix, circles, marker);

        glfwSwapBuffers(window);
        glfwPollEvents();

        int key = cv::waitKey(1);
        if (key == 27)
            break;
    }

    glfwTerminate();
    return 0;
}


void initGL(int argc, char* argv[])
{
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLfloat zero[] = { 0, 0, 0, 1.0 };
    GLfloat light_pos[] = { 0.5, 0.3, 1.0, 0 };
    GLfloat light_pos2[] = { -0.5, -0.3, -1.0, 0 };
    GLfloat light_amb[] = { 0.075, 0.075, 0.075, 1.0 };
    GLfloat light_dif[] = { 0.7, 0.7, 0.7, 1.0 };
    GLfloat light_dif2[] = { 0.075, 0.075, 0.075, 1.0 };
    GLfloat light_spe[] = { 0.8, 0.8, 0.8, 1.0 };

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_spe);

    glLightfv(GL_LIGHT1, GL_POSITION, light_pos2);
    glLightfv(GL_LIGHT1, GL_AMBIENT, zero);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_dif2);
    glLightfv(GL_LIGHT1, GL_SPECULAR, zero);

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, light_spe);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 120.0);

    glShadeModel(GL_SMOOTH);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DITHER);
    glEnable(GL_FRAMEBUFFER_SRGB);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);

    glClearColor(0.4, 0.5, 0.8, 1);
    glClearDepth(1);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glPixelZoom(1.0, -1.0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
}

void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Vec3f>& circles, bool marker)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 1, 1, 0);
    glRasterPos2i(0, 0);

    glDrawPixels(img_background.cols, img_background.rows, GL_BGR, GL_UNSIGNED_BYTE, img_background.data);

    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);

    glMatrixMode(GL_MODELVIEW);

    if (marker)
    {
        float transpose[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                transpose[i * 4 + j] = markerMatrix[j * 4 + i];

        glLoadMatrixf(transpose);
        glScalef(0.025F, 0.025F, 0.025F);
        glRotatef(-90, 1, 0, 0);

        // Draw 3 white spheres
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glTranslatef(0.0, 0.8, 0.0);
        drawSphere(0.8, 50, 50);
        glTranslatef(0.0, 0.8, 0.0);
        drawSphere(0.6, 50, 50);
        glTranslatef(0.0, 0.6, 0.0);
        drawSphere(0.4, 50, 50);

        // Draw the eyes
        glColor4f(0.0, 0.0, 0.0, 1.0);
        glPushMatrix();
        glTranslatef(-0.2, 0.2, 0.25);
        drawSphere(0.075, 20, 20);
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.2, 0.2, 0.25);
        drawSphere(0.075, 20, 20);
        glPopMatrix();

        glColor4f(1.0, 0.5, 0.0, 1.0);
        glTranslatef(0.0, 0, 0.3);
        drawCone(0.075, 0.4, 20, 20);
    }

    // z Position from last markerMatrix
    double z = markerMatrix[11];
    // For mapping 2d pixel coordinates to 3d coordinates, adjust for (downscaled) resolution and perspective
    double scale = 1.0 / (tan(fov / 180.0 * M_PI) * abs(z - 1)) / (double)img_background.rows / downscale;
    double time = glfwGetTime() *5;

    glLoadIdentity();

    for (int i = 0; i < circles.size(); ++i)
    {
        const Vec3f &v = circles[i];

        glPushMatrix();
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glTranslatef((v[0] - img_background.cols * downscale * 0.5) * scale, (img_background.rows * downscale * 0.5 - v[1]) * scale, z);
        
        if (i == 0)
            drawSphere(0.01, 20, 20);
        else
        {
            drawCylinder(0.01, 0.001, 20);
            glColor4f(1.0, 0.5, 0.0, 1.0);

            drawCylinder(0.0025, 0.03, 20);

            // drawFlag
            glRotatef(45 + cos(time * 0.25) * 12, 0, 0, 1);
            glBegin(GL_QUAD_STRIP);
            double length = 0.02;
            double height = 0.02;
            for (int s = 0; s < 40; s++)
            {
                double t = (s >= 20 ? 39 - s : s) / 20.0;
                double x = 0.002 + length * t;
                double y = (sin(t * 1 + time) - sin(time)) * 0.0025 + (sin(t * 0.5 + time * 0.25 + 0.3) - sin(time * 0.25 + 0.3)) * 0.001;
                double dy = cos(t * 1 + time) * 1 * 0.0025 + cos(t * 0.5 + time * 0.25 + 0.3) * 0.5 * 0.001;
                if (s >= 20)
                    glNormal3f(dy, 1 / 20.0 * length, 0);
                else
                    glNormal3f(-dy, -1 / 20.0 * length, 0);
                glVertex3f(x, y, 0.03 - height * 0.5 * (2 - t));
                glVertex3f(x, y, 0.03 - height * 0.5 * t);
            }
            glEnd();
        }
        
        glPopMatrix();
    }
}


void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    float ratio = width / (float)height;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(fov, ((double)width / (double)height), 0.01, 100);
}
