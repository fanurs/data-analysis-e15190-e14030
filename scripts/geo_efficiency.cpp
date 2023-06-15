/**
  * This script computes the geometric efficiency of the detector.
  * It is a standalone script that does not need to include any other custom scripts.
  * It only requires the standard library, the nlohmann/json library, and OpenMP.
  * If you are using C++ environment with ROOT, these are already included.
*/
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "omp.h"

class Vector3 {
public:
    double x, y, z;

    Vector3(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Vector3 operator*(double scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    double dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector3 normalize() const {
        double len = length();
        return Vector3(x / len, y / len, z / len);
    }

    double theta() const {
        return std::atan2(std::sqrt(x * x + y * y), z);
    }

    double phi() const {
        return std::atan2(y, x);
    }

    static Vector3 spherical_to_cartesian(double r, double theta, double phi) {
        double x = r * std::sin(theta) * std::cos(phi);
        double y = r * std::sin(theta) * std::sin(phi);
        double z = r * std::cos(theta);
        return Vector3(x, y, z);
    }
};

class Ray {
private:
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<double> uniform;

public:
    Vector3 origin, direction;

    Ray(const Vector3& direction) : origin(0, 0, 0), direction(direction.normalize()) {}

    Ray() : origin(0, 0, 0), direction(0, 0, 0) { // isotropic distribution
        double theta = std::acos(1 - 2 * this->uniform(this->gen));
        double phi = 2 * M_PI * this->uniform(this->gen);
        direction = Vector3::spherical_to_cartesian(1.0, theta, phi);
    }

    Ray(double theta) : origin(0, 0, 0), direction(0, 0, 0) { // isotropic distribution in fixed theta
        double phi = 2 * M_PI * this->uniform(this->gen);
        direction = Vector3::spherical_to_cartesian(1.0, theta, phi);
    }
};
std::random_device Ray::rd;
std::mt19937 Ray::gen(Ray::rd());
std::uniform_real_distribution<double> Ray::uniform(0.0f, 1.0);

class Cuboid {
public:
    Vector3 center;
    std::array<Vector3, 3> axes;
    std::array<double, 3> lengths;

    Cuboid(const Vector3& center, const std::array<Vector3, 3>& axes, const std::array<double, 3>& lengths)
        : center(center), axes(axes), lengths(lengths) {}

    Cuboid cut(double x_min, double x_max) {
        double x_shift = (x_max + x_min) / 2.0;
        std::array<double, 3> new_lengths = lengths;
        new_lengths[0] = x_max - x_min;
        return Cuboid(center + axes[0] * x_shift, axes, new_lengths);
    }
};

bool testAxis(const Vector3& axis, const Ray& ray, const Cuboid& cuboid) {
    double cuboidProjection = 0.0f;

    for (int i = 0; i < 3; ++i) {
        cuboidProjection += std::abs(axis.dot(cuboid.axes[i]) * 0.5 * cuboid.lengths[i]);
    }

    double rayProjection = ray.direction.dot(axis);
    double distance = std::abs(axis.dot(cuboid.center - ray.origin));

    const double epsilon = 1e-6;
    if (std::abs(rayProjection) < epsilon) {
        return distance > cuboidProjection;
    }

    double t = (distance - cuboidProjection) / rayProjection;
    return t < 0;
}

bool rayIntersectsCuboid(const Ray& ray, const Cuboid& cuboid) {
    double t_min = 0.0;
    double t_max = std::numeric_limits<double>::max();

    for (int i = 0; i < 3; ++i) {
        Vector3 slab_center = cuboid.center;
        Vector3 slab_normal = cuboid.axes[i].normalize();

        double numerator = slab_center.dot(slab_normal) - ray.origin.dot(slab_normal);
        double denominator = ray.direction.dot(slab_normal);

        // Check if the ray is parallel to the slab
        const double epsilon = 1e-6f;
        if (std::abs(denominator) < epsilon) {
            if (numerator < -0.5 * cuboid.lengths[i] || numerator > 0.5 * cuboid.lengths[i]) {
                return false;
            }
        } else {
            double t1 = (numerator - 0.5 * cuboid.lengths[i]) / denominator;
            double t2 = (numerator + 0.5 * cuboid.lengths[i]) / denominator;

            if (t1 > t2) {
                std::swap(t1, t2);
            }

            t_min = std::max(t_min, t1);
            t_max = std::min(t_max, t2);

            if (t_min > t_max) {
                return false;
            }
        }
    }

    return true;
}

class NeutronWall {
private:
    char AB;
    bool include_pyrex;
    double length_x_cm, length_y_cm, length_z_cm;
    std::map<int, Cuboid> barCuboids;

    std::filesystem::path getFilePath() {
        std::filesystem::path path(std::getenv("PROJECT_DIR"));
        path /= "database/neutron_wall/geometry/NW" + std::string(1, AB) + "_pca.dat";
        return path;
    }

    void createBarCuboidsFromFile() {
        std::map<int, Vector3> bar_centers;
        std::map<int, std::array<Vector3, 3> > axes;

        std::filesystem::path path = getFilePath();
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (line.find('#') == 0) continue;

            std::istringstream iss(line);
            int bar_num;
            std::string vector_type;
            double x, y, z;
            if (!(iss >> bar_num >> vector_type >> x >> y >> z)) break;
            if (vector_type == "L") {
                bar_centers[bar_num] = Vector3(x, y, z);
            } else {
                Vector3 vector(x, y, z);
                if (axes.find(bar_num) == axes.end()) { // if not found
                    axes[bar_num] = std::array<Vector3, 3>{};
                }
                if (vector_type == "X") {
                    axes[bar_num][0] = vector.normalize();
                } else if (vector_type == "Y") {
                    axes[bar_num][1] = vector.normalize();
                } else if (vector_type == "Z") {
                    axes[bar_num][2] = vector.normalize();
                }
            }
        }
        file.close();

        for (auto& [bar_num, center] : bar_centers) {
            this->barCuboids.emplace(
                bar_num,
                Cuboid(center, axes[bar_num], { this->length_x_cm, this->length_y_cm, this->length_z_cm })
            );
        }
    }

    void populateCuboids(nlohmann::json& filters) {
        // populate bar_ranges from filters
        std::map<int, std::vector< std::array<double, 2> > > bar_ranges;
        for (auto& [bar_num, ranges] : filters.items()) {
            const double min_range = -0.5 * this->barCuboids.at(std::stoi(bar_num)).lengths[0];
            const double max_range = +0.5 * this->barCuboids.at(std::stoi(bar_num)).lengths[0];

            for (auto& range : ranges) {
                // range outside of cuboid will be truncated
                double range_start = std::max(range[0].get<double>(), min_range);
                double range_stop = std::min(range[1].get<double>(), max_range);
                if (range_start < range_stop) {
                    bar_ranges[std::stoi(bar_num)].push_back({ range_start, range_stop });
                }
            }
        }

        for (auto& [bar_num, barCuboid] : barCuboids) {
            for (auto& [x_min, x_max] : bar_ranges[bar_num]) {
                this->cuboids.push_back(barCuboid.cut(x_min, x_max));
            }
        }
    }

public:
    std::vector<Cuboid> cuboids;

    NeutronWall(
        char AB, nlohmann::json& wall_filters, bool include_pyrex=false
    ) : AB(AB), include_pyrex(include_pyrex) {
        this->length_x_cm = 76; // inches
        this->length_y_cm = 3; // inches
        this->length_z_cm = 2.5; // inches
        if (include_pyrex) {
            this->length_x_cm += 0.25; // inches
            this->length_y_cm += 0.25; // inches
            this->length_z_cm += 0.25; // inches
        }
        this->length_x_cm *= 2.54; // cm
        this->length_y_cm *= 2.54; // cm
        this->length_z_cm *= 2.54; // cm

        this->createBarCuboidsFromFile();
        this->populateCuboids(wall_filters);
    }
};

double findIntersectingPoint(
    double phi_min, double phi_max, double theta, const NeutronWall& wall, bool find_lower_bound, double tolerance=1e-5
) {
    double low = phi_min;
    double high = phi_max;
    double mid;

    while (high - low > tolerance) {
        mid = (low + high) / 2;
        Ray ray(Vector3::spherical_to_cartesian(1.0, theta, mid));
        bool intersection_found = false;
        for (auto& cuboid : wall.cuboids) {
            if (rayIntersectsCuboid(ray, cuboid)) {
                intersection_found = true;
                break;
            }
        }

        if (intersection_found == find_lower_bound) high = mid;
        else low = mid;
    }
    return (low + high) / 2.0;
}

double getGeometricEfficiencyUsingMonteCarlo(const double theta, NeutronWall& wall, int num_rays) {
    int num_intersections = 0;
    #pragma omp parallel for reduction(+:num_intersections)
    for (int i = 0; i < num_rays; ++i) {
        Ray ray(theta);
        bool intersection_found = false;
        for (auto& cuboid : wall.cuboids) {
            if (rayIntersectsCuboid(ray, cuboid)) {
                intersection_found = true;
                break;
            }
        }
        if (intersection_found) num_intersections++;
    }
    return 1.0 * num_intersections / num_rays;
}

double getGeometricEfficiencyUsingDeltaPhi(const double theta, NeutronWall& wall) {
    std::vector< std::pair<double, double> > intersected_ranges;
    const double phi_step = 1e-4 * M_PI;
    bool range_open = false;
    for (double phi = -M_PI; phi < M_PI; phi += phi_step) {
        Ray ray(Vector3::spherical_to_cartesian(1.0, theta, phi));
        bool intersection_found = false;
        for (auto& cuboid : wall.cuboids) {
            if (rayIntersectsCuboid(ray, cuboid)) {
                intersection_found = true;
                break;
            }
        }

        if (intersection_found) {
            if (!range_open || intersected_ranges.empty()) {
                intersected_ranges.push_back({phi, phi});
                range_open = true;
            } else { // keep extending the range
                intersected_ranges.back().second = phi;
            }
        } else {
            if (range_open) { // close the range
                intersected_ranges.back().second = phi;
                range_open = false;
            }
        }
    }

    double total_phi_range = 0;
    for (auto& [phi_min, phi_max] : intersected_ranges) {
        double refined_phi_min = findIntersectingPoint(phi_min - 2 * phi_step, phi_min + 2 * phi_step, theta, wall, true);
        double refined_phi_max = findIntersectingPoint(phi_max - 2 * phi_step, phi_max + 2 * phi_step, theta, wall, false);
        total_phi_range += refined_phi_max - refined_phi_min;
    }

    return total_phi_range / (2 * M_PI);
}

/**
 * Get the geometric efficiency of a wall at a fixed theta angle.
 *
 * @param wall_filters_str Wall filters in JSON format, e.g. {"1": [[-90, 90]], "2":
 * [[-90, -20], [20, 90]]}, where the keys are the bar numbers and the values
 * are the position x ranges in cm.
 * @param theta Theta in degree.
 * @param num_rays Number of rays to use for Monte Carlo. If 0, delta-phi method. Default 0.
**/
double getGeometryEfficiency(char AB, bool include_pyrex, const std::string& wall_filters_str, double theta, int num_rays=0) {
    nlohmann::json wall_filters = nlohmann::json::parse(wall_filters_str);
    NeutronWall wall(AB, wall_filters, include_pyrex);
    if (num_rays > 0) { // Monte Carlo
        return getGeometricEfficiencyUsingMonteCarlo(theta, wall, num_rays);
    } else { // delta-phi method
        return getGeometricEfficiencyUsingDeltaPhi(theta, wall);
    }
}

int main(int argc, const char* argv[]) {
    // parse arguments
    std::string method = (argv[1][0] == 'm') ? "monte_carlo" : "delta_phi";
    char AB = std::toupper(argv[2][0]);
    bool include_pyrex = std::stoi(argv[3]);
    std::string wall_filters_str = argv[4];

    // arguments that depend on the method
    std::string mode; // "single" or "range"
    double theta;
    int num_rays = 0;
    double theta_low, theta_upp;
    int n_steps;
    if (method == "monte_carlo") {
        theta = std::atof(argv[5]) * M_PI / 180.0;
        num_rays = std::atoi(argv[6]);
        mode = "single";
    } else if (method == "delta_phi" && argc - 1 == 5) {
        theta = std::atof(argv[5]) * M_PI / 180.0;
        mode = "single";
    } else if (method == "delta_phi" && argc - 1 > 5) {
        theta_low = std::atof(argv[5]) * M_PI / 180.0;
        theta_upp = std::atof(argv[6]) * M_PI / 180.0;
        n_steps = std::stoi(argv[7]);
        mode = "range";
    }

    // get geometry efficiency and print to stdout
    if (method == "monte_carlo") {
        double geo_eff = getGeometryEfficiency(AB, include_pyrex, wall_filters_str, theta, num_rays);
        std::cout << std::fixed << std::setprecision(10) << geo_eff << std::endl;
    } else if (method == "delta_phi" && mode == "single") {
        double geo_eff = getGeometryEfficiency(AB, include_pyrex, wall_filters_str, theta);
        std::cout << std::fixed << std::setprecision(10) << geo_eff << std::endl;
    } else if (method == "delta_phi" && mode == "range") {
        std::vector<double> results(n_steps);
        #pragma omp parallel for
        for (int i = 0; i < n_steps; i++) {
            double theta = theta_low + (theta_upp - theta_low) * i / (n_steps - 1); // like numpy.linspace
            results[i] = getGeometryEfficiency(AB, include_pyrex, wall_filters_str, theta);
        }

        for (int i = 0; i < n_steps; i++) {
            std::cout << std::fixed << std::setprecision(10) << results[i] << std::endl;
        }
    }

    return 0;
}
