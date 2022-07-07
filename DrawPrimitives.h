#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <math.h>


/* PI */
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif


void drawSphere(double r, int lats, int longs) {
	int i, j;
	for (i = 0; i <= lats; i++) {
		double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
		double z0 = r * sin(lat0);
		double zr0 = r * cos(lat0);

		double lat1 = M_PI * (-0.5 + (double)i / lats);
		double z1 = r * sin(lat1);
		double zr1 = r * cos(lat1);

		glBegin(GL_QUAD_STRIP);
		for (j = 0; j <= longs; j++) {
			double lng = 2 * M_PI * (double)(j - 1) / longs;
			double x = cos(lng);
			double y = sin(lng);

			glNormal3f(x * zr0, z0, y * zr0);
			glVertex3f(x * zr0, z0, y * zr0);
			glNormal3f(x * zr1, z1, y * zr1);
			glVertex3f(x * zr1, z1, y * zr1);
		}
		glEnd();
	}
}


void drawCone(GLdouble base, GLdouble height, GLint slices, GLint stacks)
{

	// draw the upper part of the cone
	glBegin(GL_TRIANGLES);
	for (int s = 0; s < slices; s++) {
		double angle = (s + 1.0) / (double)slices * M_PI * 2;
		double x = sin((double)angle);
		double y = cos((double)angle);
		glNormal3f(x * height, y * height, base);
		glVertex3f(x * base, y * base, 0.f);
		angle = s / (double)slices * M_PI * 2;
		x = sin((double)angle);
		y = cos((double)angle);
		glNormal3f(x * height, y * height, base);
		glVertex3f(x * base, y * base, 0.f);
		angle = (s + 0.5) / (double)slices * M_PI * 2;
		x = sin((double)angle);
		y = cos((double)angle);
		glNormal3f(x * height, y * height, base);
		glVertex3f(0, 0, height);
	}
	glEnd();

	// draw the base of the cone
	glBegin(GL_TRIANGLE_FAN);
	glNormal3f(0, -1, 0);
	glVertex3f(0, 0, 0);
	for (int s = 0; s <= slices; s++) {
		double angle = s / (double)slices * M_PI * 2;
		glVertex3f(sin((double)angle) * base, cos((double)angle) * base, 0.f);
	}
	glEnd();
}

void drawCylinder(GLdouble base, GLdouble height, GLint slices)
{

	glBegin(GL_QUAD_STRIP);
	for (int s = 0; s <= slices; s++) {
		double angle = s / (double)slices * M_PI * 2;

		double x = sin((double)angle);
		double y = cos((double)angle);
		glNormal3f(x, y, 0);
		glVertex3f(x * base, y * base, 0.f);
		glVertex3f(x * base, y * base, height);
	}
	glEnd();

	// draw the base of the cylinder
	glBegin(GL_TRIANGLE_FAN);
	glNormal3f(0, 0, -1);
	glVertex3f(0, 0, 0);
	for (int s = 0; s <= slices; s++) {
		double angle = s / (double)slices * M_PI * 2;
		glVertex3f(sin((double)angle) * base, cos((double)angle) * base, 0.f);
	}
	glEnd();

	// draw the top of the cone
	glBegin(GL_TRIANGLE_FAN);
	glNormal3f(0, 0, 1);
	glVertex3f(0, 0, height);
	for (int s = 0; s <= slices; s++) {
		double angle = -s / (double)slices * M_PI * 2;
		glVertex3f(sin((double)angle) * base, cos((double)angle) * base, height);
	}
	glEnd();
}
