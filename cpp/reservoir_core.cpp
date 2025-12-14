#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

struct Bounds {
    std::array<double,5> lo{500, 500, 500, 500, 2000};
    std::array<double,5> hi{6000,6000,6000,6000,15000};
} g_bounds;

static inline std::array<double,5> to_unit(const std::array<double,5>& raw) {
    std::array<double,5> z{};
    for (int i=0;i<5;i++) {
        z[i] = (raw[i] - g_bounds.lo[i]) / (g_bounds.hi[i] - g_bounds.lo[i]);
        z[i] = clamp(z[i], 0.0, 1.0);
    }
    return z;
}

static inline std::array<double,5> to_raw(const std::array<double,5>& z) {
    std::array<double,5> raw{};
    for (int i=0;i<5;i++) {
        double v = clamp(z[i], 0.0, 1.0);
        raw[i] = g_bounds.lo[i] + v * (g_bounds.hi[i] - g_bounds.lo[i]);
    }
    return raw;
}

struct SimResult {
    double total_prod=0, total_inj=0;
    double P_min=0, P_max=0, Pi=0;
    double R_prod=0, R_pres=0, R_inj=0;
    double S=0;
};

SimResult simulate_reservoir(const std::array<double,5>& oper_con,
                            int T=60,
                            double Pi=3000.0,
                            double Pcrit=1500.0,
                            double k_prod=0.00015,
                            double k_inj =0.00008,
                            double leak  =0.015)
{
    double qsum = oper_con[0]+oper_con[1]+oper_con[2]+oper_con[3];
    double inj  = oper_con[4];

    double P = Pi;
    double Pmin = P, Pmax = P;

    double total_prod = 0.0;
    double total_inj  = 0.0;

    for (int t=0;t<T;t++) {
        double dP = -k_prod*qsum + k_inj*inj - leak*(P - Pi);
        P += dP;
        Pmin = std::min(Pmin, P);
        Pmax = std::max(Pmax, P);

        total_prod += qsum;
        total_inj  += inj;
    }

    double prod_ref = 4.0 * 4000.0 * T;
    double R_prod = clamp(total_prod / prod_ref, 0.0, 1.5);

    double R_pres = 1.0;
    if (Pmin < Pcrit) {
        double x = (Pcrit - Pmin) / 300.0;
        R_pres = std::exp(-x);
    }

    double inj_ref = 9000.0 * T;
    double inj_ratio = (inj_ref > 0 ? total_inj / inj_ref : 1.0);
    double R_inj = std::exp(-0.25 * std::max(0.0, inj_ratio - 1.0));

    double S = clamp(R_prod * R_pres * R_inj, 0.0, 10.0);

    SimResult res;
    res.total_prod = total_prod;
    res.total_inj  = total_inj;
    res.P_min = Pmin;
    res.P_max = Pmax;
    res.Pi = Pi;
    res.R_prod = R_prod;
    res.R_pres = R_pres;
    res.R_inj  = R_inj;
    res.S = S;
    return res;
}

static double halton(int index, int base) {
    double f = 1.0;
    double r = 0.0;
    int i = index;
    while (i > 0) {
        f = f / base;
        r = r + f * (i % base);
        i = i / base;
    }
    return r;
}

static inline uint64_t pack_cell(int c0,int c1,int c2,int c3,int c4){
    return (uint64_t(c0)      ) |
           (uint64_t(c1) << 12) |
           (uint64_t(c2) << 24) |
           (uint64_t(c3) << 36) |
           (uint64_t(c4) << 48);
}

struct Ascent {
    int lo = -1;
    int hi = -1;
    double slope = -1.0;
};

static inline double dist5(const std::array<double,5>& a, const std::array<double,5>& b){
    double s=0;
    for(int i=0;i<5;i++){ double d=a[i]-b[i]; s+=d*d; }
    return std::sqrt(s);
}

static inline std::array<double,5> local_sample_along(
    const std::array<double,5>& z0,
    const std::array<double,5>& z1,
    std::mt19937& rng,
    double alpha_max_scale = 2.0,
    double noise_radius    = 0.05
){
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Unoise(-1.0, 1.0);

    std::array<double,5> d{};
    for(int i=0;i<5;i++) d[i] = (z1[i]-z0[i]);
    double dn = 0;
    for(int i=0;i<5;i++) dn += d[i]*d[i];
    dn = std::sqrt(std::max(dn, 1e-12));
    for(int i=0;i<5;i++) d[i] /= dn;

    double alpha_max = alpha_max_scale * dn;
    double alpha = U01(rng) * alpha_max;

    std::array<double,5> z = z0;
    for(int i=0;i<5;i++){
        double n = Unoise(rng) * noise_radius;
        z[i] = clamp(z[i] + alpha*d[i] + n, 0.0, 1.0);
    }
    return z;
}

struct Sample { std::array<double,5> z; double S; };

static py::tuple generate_samples_cpp(int N_global=50000, int M_ascents=100, int N_local_per_ascent=3000, int B=32) {
    py::gil_scoped_release release;

    const int bases[5] = {2,3,5,7,11};
    std::vector<Sample> samples;
    samples.resize(N_global);

    #pragma omp parallel for if(N_global>20000)
    for(int i=0;i<N_global;i++){
        int idx = i + 1;
        std::array<double,5> z{};
        for(int d=0; d<5; d++) z[d] = halton(idx, bases[d]);
        auto raw = to_raw(z);
        auto res = simulate_reservoir(raw);
        samples[i] = {z, res.S};
    }

    auto cell_of = [&](const std::array<double,5>& z){
        std::array<int,5> c{};
        for(int i=0;i<5;i++){
            int ci = int(std::floor(z[i] * B));
            if(ci < 0) ci = 0;
            if(ci >= B) ci = B-1;
            c[i] = ci;
        }
        return c;
    };

    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(size_t(N_global)*2);

    for(int i=0;i<N_global;i++){
        auto c = cell_of(samples[i].z);
        uint64_t key = pack_cell(c[0],c[1],c[2],c[3],c[4]);
        grid[key].push_back(i);
    }

    std::vector<Ascent> ascents;
    ascents.reserve(N_global);

    for(int i=0;i<N_global;i++){
        const auto& zi = samples[i].z;
        const double Si = samples[i].S;
        auto ci = cell_of(zi);

        int best_j = -1;
        double best_slope = -1.0;

        for(int d0=-1; d0<=1; d0++)
        for(int d1=-1; d1<=1; d1++)
        for(int d2=-1; d2<=1; d2++)
        for(int d3=-1; d3<=1; d3++)
        for(int d4=-1; d4<=1; d4++){
            int c0 = ci[0]+d0, c1=ci[1]+d1, c2=ci[2]+d2, c3=ci[3]+d3, c4=ci[4]+d4;
            if(c0<0||c1<0||c2<0||c3<0||c4<0||c0>=B||c1>=B||c2>=B||c3>=B||c4>=B) continue;
            uint64_t key = pack_cell(c0,c1,c2,c3,c4);
            auto it = grid.find(key);
            if(it==grid.end()) continue;

            for(int j : it->second){
                if(j==i) continue;
                double Sj = samples[j].S;
                if(Sj <= Si) continue;

                double d = dist5(zi, samples[j].z);
                if(d < 1e-9) continue;
                double slope = (Sj - Si) / d;

                if(slope > best_slope){
                    best_slope = slope;
                    best_j = j;
                }
            }
        }

        if(best_j >= 0 && best_slope > 0){
            ascents.push_back({i, best_j, best_slope});
        }
    }

    int useM = std::min(M_ascents, (int)ascents.size());
    if(useM > 0){
        std::nth_element(ascents.begin(),
                         ascents.begin() + useM,
                         ascents.end(),
                         [](const Ascent& a, const Ascent& b){ return a.slope > b.slope; });
        ascents.resize(useM);
    }

    int start_idx = (int)samples.size();
    samples.resize(N_global + useM * N_local_per_ascent);

    #pragma omp parallel for if(useM>4)
    for(int k=0; k<useM; k++){
        std::mt19937 rng(1337 + 1000*k);
        const auto& z0 = samples[ascents[k].lo].z;
        const auto& z1 = samples[ascents[k].hi].z;

        for(int t=0; t<N_local_per_ascent; t++){
            int out_i = start_idx + k*N_local_per_ascent + t;
            auto z = local_sample_along(z0, z1, rng, 2.0, 0.04);
            auto raw = to_raw(z);
            auto res = simulate_reservoir(raw);
            samples[out_i] = {z, res.S};
        }
    }

    py::gil_scoped_acquire acquire;

    const ssize_t N = (ssize_t)samples.size();
    py::array_t<double> Z({N, 5});
    py::array_t<double> S({N});

    auto Zm = Z.mutable_unchecked<2>();
    auto Sm = S.mutable_unchecked<1>();

    for(ssize_t i=0;i<N;i++){
        for(int d=0; d<5; d++) Zm(i,d) = samples[i].z[d];
        Sm(i) = samples[i].S;
    }

    return py::make_tuple(Z, S, useM);
}

static std::array<double,5> arr5_from_np(const py::array_t<double>& a){
    auto b = a.unchecked<1>();
    if(b.shape(0) != 5) throw std::runtime_error("expected shape (5,)");
    std::array<double,5> x{};
    for(int i=0;i<5;i++) x[i] = b(i);
    return x;
}

static py::array_t<double> np_from_arr5(const std::array<double,5>& x){
    py::array_t<double> a({5});
    auto m = a.mutable_unchecked<1>();
    for(int i=0;i<5;i++) m(i) = x[i];
    return a;
}

static py::dict simulate_py(const py::array_t<double>& raw_np){
    auto raw = arr5_from_np(raw_np);
    auto r = simulate_reservoir(raw);
    py::dict d;
    d["total_prod"] = r.total_prod;
    d["total_inj"]  = r.total_inj;
    d["P_min"]      = r.P_min;
    d["P_max"]      = r.P_max;
    d["Pi"]         = r.Pi;
    d["R_prod"]     = r.R_prod;
    d["R_pres"]     = r.R_pres;
    d["R_inj"]      = r.R_inj;
    d["S"]          = r.S;
    return d;
}

PYBIND11_MODULE(reservoir_core, m) {
    m.def("generate_samples", &generate_samples_cpp,
          py::arg("N_global")=50000, py::arg("M_ascents")=100, py::arg("N_local_per_ascent")=3000, py::arg("B")=32);

    m.def("to_raw", [](const py::array_t<double>& z){ return np_from_arr5(to_raw(arr5_from_np(z))); });
    m.def("to_unit", [](const py::array_t<double>& raw){ return np_from_arr5(to_unit(arr5_from_np(raw))); });

    m.def("simulate", &simulate_py);
}
