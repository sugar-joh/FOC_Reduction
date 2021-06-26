// Compute the error from the uncertainties in the direction of polarizer's axes

#include <iostream>
#include <math.h>


using namespace std;

const double PI=3.14159265359;	

const double I = 8.42755038425555;
const double Q = -0.7409864902131016;
const double U = -1.079689321440609;

const double k1 = 0.986;			//from Kishimoto (1999)
const double k2 = 0.976;			//from Kishimoto (1999)
const double k3 = 0.973;			//from Kishimoto (1999)

const double Theta1 = 180*PI/180.;		//radians
const double Theta2 = 60*PI/180.;		//radians
const double Theta3 = 120*PI/180.;		//radians

const double sigma_theta_1=3*PI/180.;		//radians
const double sigma_theta_2=3*PI/180.;		//radians
const double sigma_theta_3=3*PI/180.;		//radians



double A()		
{
 	return k2*k3*sin(-2*Theta2+2*Theta3) + k3*k1*sin(-2*Theta3+2*Theta1) + k1*k2*sin(-2*Theta1+2*Theta2);
}


double Ap(int i)		//A'_i
{
	if(i==1) return 2*k3*k1*cos(-2*Theta3+2*Theta1) - 2*k1*k2*cos(-2*Theta1+2*Theta2);
	if(i==2) return -2*k2*k3*cos(-2*Theta2+2*Theta3) + 2*k1*k2*cos(-2*Theta1+2*Theta2);
	if(i==3) return 2*k2*k3*cos(-2*Theta2+2*Theta3) - 2*k3*k1*cos(-2*Theta3+2*Theta1);
	else return 0;
}


double Flux(int i)		//f'_i
{
	if(i==1) return I-(k1*cos(2*Theta1))*Q-(k1*sin(2*Theta1))*U;
	if(i==2) return I-(k2*cos(2*Theta2))*Q-(k2*sin(2*Theta2))*U;
	if(i==3) return I-(k3*cos(2*Theta3))*Q-(k3*sin(2*Theta3))*U;
	else return 0;
}


double coef(int i, int j)	//a_ij
{
	if(i==1 && j==1) return (1./A())*(k2*k3*sin(-2*Theta2+2*Theta3));
	if(i==1 && j==2) return (1./A())*(k3*k1*sin(-2*Theta3+2*Theta1));
	if(i==1 && j==3) return (1./A())*(k1*k2*sin(-2*Theta1+2*Theta2));
	if(i==2 && j==1) return (1./A())*(-k2*sin(2*Theta2)+k3*sin(2*Theta3));
	if(i==2 && j==2) return (1./A())*(-k3*sin(2*Theta3)+k1*sin(2*Theta1));
	if(i==2 && j==3) return (1./A())*(-k1*sin(2*Theta1)+k2*sin(2*Theta2));
	if(i==3 && j==1) return (1./A())*(k2*cos(2*Theta2)-k3*cos(2*Theta3));
	if(i==3 && j==2) return (1./A())*(k3*cos(2*Theta3)-k1*cos(2*Theta1));
	if(i==3 && j==3) return (1./A())*(k1*cos(2*Theta1)-k2*cos(2*Theta2));
	else return 0;	
}


double g()		// I_2 == Q
{
	double a,b,c;

	a=coef(2,1)*Flux(1);
	b=coef(2,2)*Flux(2);
	c=coef(2,3)*Flux(3);
	
 	return a+b+c;
}


double g_p()	// I'_2 == Q'
{
	double f=coef(2,1)*Flux(1)+coef(2,2)*Flux(2)+coef(2,3)*Flux(3);
	double fp=2*k2*cos(2*Theta2)*(k1*Q*cos(2*Theta1)+k1*U*sin(2*Theta1)-I)+(k1*sin(2*Theta1)-k3*sin(2*Theta3))*(2*k2*Q*sin(2*Theta2)-2*k2*U*cos(2*Theta2))-2*k2*cos(2*Theta2)*(k3*Q*cos(2*Theta3)+k3*U*sin(2*Theta3)-I);
	return (A()+fp-f*Ap(2))/(pow(A(),2));
}

double h()		// I_3 == U
{
	double a,b,c;

	a=coef(3,1)*Flux(1);
	b=coef(3,2)*Flux(2);
	c=coef(3,3)*Flux(3);
	
 	return a+b+c;
}


double h_p()	// I'_3 == U'
{
	double f=coef(3,1)*Flux(1)+coef(3,2)*Flux(2)+coef(3,3)*Flux(3);
	double fp=-2*k3*sin(2*Theta3)*(k1*Q*cos(2*Theta1)+k1*U*sin(2*Theta1)-I)+2*k3*sin(2*Theta3)*(k2*Q*cos(2*Theta2)+k2*U*sin(2*Theta2)-I)+2*k3*(k2*cos(2*Theta2)-k1*cos(2*Theta1))*(U*cos(2*Theta3)-Q*sin(2*Theta3));
	return (A()+fp-f*Ap(3))/(pow(A(),2));
}


double k()		// I_1 == I
{
	double a,b,c;

	a=coef(1,1)*Flux(1);
	b=coef(1,2)*Flux(2);
	c=coef(1,3)*Flux(3);
	
 	return a+b+c;
}


double k_p()	// I'_1 == I'
{
	double f=coef(1,1)*Flux(1)+coef(1,2)*Flux(2)+coef(1,3)*Flux(3);
	double fp=2*k1*k2*k3*sin(2*Theta2-2*Theta3)*(U*cos(2*Theta1)-Q*sin(2*Theta1))-2*k1*k3*cos(2*Theta1-2*Theta3)*(k2*Q*cos(2*Theta2)+k2*U*sin(2*Theta2)-I)+2*k1*k2*cos(2*Theta1-2*Theta2)*(k3*Q*cos(2*Theta3)+k3*U*sin(2*Theta3)-I);
	return (A()+fp-f*Ap(1))/(pow(A(),2));
}


double dTheta_i(int i)		
{
	double a,b,c;

	if(i==1)			//partial derivative with i=1
	{
		a = g()*g_p();
		b = k();
		c = sqrt(pow(g(),2)+pow(h(),2));
		return a/(b*c);
	}
	if(i==2)			//partial derivative with i=2
	{
		a = h()*h_p();
		b = k();
		c = sqrt(pow(g(),2)+pow(h(),2));
		return a/(b*c);
	}
	if(i==3)			//partial derivative with i=3
	{
		a = k_p();
		b = sqrt(pow(g(),2)+pow(h(),2));
		c = pow(k(),2);
		return -a*b/c;
	}
	else return 0;
}



int main()
{
	double sigma_stat_P;
	sigma_stat_P = 0;

	sigma_stat_P=pow(dTheta_i(1),2)*pow(sigma_theta_1,2)+pow(dTheta_i(2),2)*pow(sigma_theta_2,2)+pow(dTheta_i(3),2)*pow(sigma_theta_3,2);

	cout << "P: " << sqrt(Q*Q+U*U)/I << " +/- " << sqrt(sigma_stat_P) << endl;

	return 0;
}




