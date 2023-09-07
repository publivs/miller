#include <iostream>
#include <cmath>
#include <functional>
double ytm_nolast(double PV, double C, double freq, double d, double n, double M, double TS, double* cf = nullptr) {
    if (cf == nullptr) {
        static double default_cf = C / freq;
        cf = &default_cf;
    }
    
    if (TS == 0) {
        std::cerr << "TS为0，无法解出" << std::endl;
        return std::nan("");
    }
    
    std::function<double(double)> f_d = [=](double y) {
        double sum = 0.0;
        for (int t = 0; t < n; t++) {
            sum += cf[t] * std::pow(1 + y / freq, -((d / TS) + t));
        }
        return sum + M * std::pow(1 + y / freq, -((d / TS) + n - 1)) - PV;
    };
    
    std::function<double(double)> f_d_list = [=](double y) {
        double sum = 0.0;
        for (int t = 0; t < n; t++) {
            sum += cf[t] * std::pow(1 + y, -(d / TS));
        }
        return sum - PV;
    };
    
    std::function<double(double)> f = d == static_cast<int>(d) ? f_d : f_d_list;
    
    std::function<double(double)> f_diff = [=](double y) {
        double delta_y = 0.000001;
        return (f(y + delta_y) - f(y - delta_y)) / (2 * delta_y);
    };
    
    double y_guess = C / 100.0;
    int maxiter = 50;
    double tol = 1e-8;
    for (int i = 0; i < maxiter; i++) {
        double y_old = y_guess;
        y_guess = y_guess - f(y_guess) / f_diff(y_guess);
        if (std::abs(y_guess - y_old) < tol) {
            return y_guess;
        }
    }
    
    std::cerr << "求解出现问题，请进行验证" << std::endl;
    return std::nan("");
}
int main() {
    double PV = 101.6781;
    double C = 3.29;
    double freq = 2.0;
    double d = 3.0;
    double n = 18.0;
    double M = 100.0;
    double TS = 184.0;
    
    double cf[18] = {3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 103.29};
    
    double ytm = ytm_nolast(PV, C, freq, d, n, M, TS, cf);
    std::cout << "定息债券的到期收益率为：" << ytm * 100 << "%" << std::endl;
    
    return 0;
}