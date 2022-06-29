#include <iostream>
#include <GLFW/glfw3.h>
#include <DrawPrimitives.h>
#include <MarkerTracker.h>
#include <CoinTracker.h>
#include <PoseEstimation.h>
#include <opencv\cv.hpp>


using namespace cv;
using namespace std;

//callback for resizing the window
void reshape(GLFWwindow* window, int width, int height);
//openGL initialization
void initGL(int argc, char* argv[]);
//rendering
void display(GLFWwindow* window);

int main(int argc, char* argv[])
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, reshape);


    // Make the window's context current
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    initGL(argc, argv);

    //Set up OpenCV
    cv::VideoCapture cap("CoinSoccerMovie.mp4");
    
    if (!cap.isOpened())
    {
        std::cout << "No capture" << std::endl;
        return -1;
    }
    
    cv::Mat img_background;


    //initVideoStream(cap);
    const double kMarkerSize = 0.045;
    MarkerTracker markerTracker(kMarkerSize);
    CoinTracker coinTracker(6);
    std::vector<Vec3f> circles;

    float resultMatrix[16];
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window) && cap.read(img_background))
    {
        Mat resized;

        cv::resize(img_background, resized, Size(), 0.25, 0.25, CV_INTER_LINEAR);
        coinTracker.findCoins(resized, circles);
        markerTracker.findMarker(img_background, resultMatrix);

        //glClear(GL_COLOR_BUFFER_BIT);

        //Render here
        display(window);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();


    }

    //free the memory
    glfwTerminate();
    return 0;
}


void initGL(int argc, char* argv[])
{

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glPixelZoom(1.0, -1.0);

    //set background color
    glClearColor(0.0, 0.4, 0.5, 1.0);

    //enable depth test
    glEnable(GL_DEPTH_TEST);



    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);

    //set the depth value back to 1 when there was a swap
    glClearDepth(1);

    //enable face culling
    //glEnable(GL_CULL_FACE);

}

void display(GLFWwindow* window)
{


    float ratio;
    int width, height;

    glfwGetFramebufferSize(window, &width, &height);
    ratio = width / (float)height;

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    int fov = 30;
    float near = 0.01f;
    float far = 100.0f;
    float top = tan((double)(fov * M_PI / 360.0f)) * near;
    float bottom = -top;
    float left = ratio * bottom;
    float right = ratio * top;
    glFrustum(left, right, bottom, top, near, far);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //Camera is automatically positioned in (0,0,0):
    //  Move the objects backwards so that they can be seen by the camera
    glTranslatef(0.0f, -0.5f, -10.0f);

    //rotation animation
    glRotatef((float)50 * glfwGetTime(), 0.0f, 1.0f, 0.0f);

    // Draw 3 white spheres       
    glColor4f(1.0, 1.0, 1.0, 1.0);
    drawSphere(0.8, 50, 50);
    glTranslatef(0.0f, 1.0f, 0.0f);
    drawSphere(0.6, 50, 50);
    glTranslatef(0.0f, 0.8f, 0.0f);
    drawSphere(0.4, 50, 50);


    // Draw the eyes
    // Push -> save the pose (in a modelview matrix)
    glPushMatrix();

    glColor4f(0.0, 0.0, 0.0, 1.0);
    glTranslatef(0.15, 0.15, 0.35);
    drawSphere(0.05, 10, 10);

    // Pop -> go back to the last saved pose (head position)
    glPopMatrix();
    glPushMatrix();
    glTranslatef(-0.15, 0.15, 0.35);
    drawSphere(0.05, 10, 10);

    //head position
    glPopMatrix();
    //glPushMatrix();
    // Draw a nose
    glColor4f(1.0, 0.5, 0.0, 1.0);
    glRotatef(180.0f, 0.0f, 0.0f, 1.0f);
    drawCone(0.2f, 0.8f, 0, 0);

    //ligthing
    GLfloat lightPos[] = { -2.0f, -2.0f, 2.0f, 0.0f };
    GLfloat ambientColor[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    GLfloat diffuseColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };

    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientColor);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseColor);


}


void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float ratio = (GLfloat)width / (GLfloat)height;
    int fov = 30;
    float near = 0.01f;
    float far = 100.0f;
    float top = tan((double)(fov * M_PI / 360.0f)) * near;
    float bottom = -top;
    float left = ratio * bottom;
    float right = ratio * top;
    glFrustum(left, right, bottom, top, near, far);
}