#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
//#include <omp.h> // For multithreading

using namespace std;
//namespace py = pybind11;



float R = 100000; // Escape radius of the sequence
float R2 = R * R; // Used all throughout the code
int skip = 2;   // The first two iterations will be removed from the orbit calculations

pair<vector<double>, vector<double>> make_mandel(int N, int n_iter, vector<double> coords, float stripe_density)
/*
* Main function. Returns the image of the orbit by the transformation function and the basic iterations
*
* N      -> (int) Size of canvas in pixels
* n_iter -> (int) Number of maximum iterations per pixel
* coords -> (double vector) Vector of the complex coordinates [lower_x, lower_y, upper_x, upper_y]
* stripe_density -> (double) Parameter in the function applied to the orbit
*/
{

    double lx = coords[0];
    double ly = coords[1];
    double ux = coords[2];
    double uy = coords[3];

    int i;
    int k;
    int j;


    vector<double> re_Z(N * N, 0.0);
    vector<double> im_Z(N * N, 0.0);
    vector<double> iters(N * N, 0);
    vector<double> mix(N * N, 0.0);
    vector<double> re_C(N, 0.0);
    vector<double> im_C(N, 0.0);


    for (k = 0; k < N; k++)
    {
        re_C[k] = lx + (ux - lx) / N * (k + 0.5);
        im_C[k] = ly + (uy - ly) / N * (k + 0.5);
    }

    i = 0;
    for (k = 0; k < N; k++) // Initializing Z
    {
        for (j = 0; j < N; j++)
        {
            re_Z[i] = re_C[j];
            im_Z[i] = im_C[k];
            i++;

        }
    }

    int id;
    double re_sq;
    double im_sq;
    double last_added = 0;
    double stripe;
    double frac;
    double count;
    double previous_stripe;

    for (k = 0; k < N; k++)
    {
        // Important line: Allows for the loop below to be computed in parallel. Each of the CPU's logical core will calculate a different "j" at the same time.
        // You have to specify which variables should be shared (like the output vectors) and which are "private" to each loop.
//#pragma omp parallel for default(none) shared(mix, iters, re_Z, im_Z, re_C, im_C) firstprivate(k) private(j,re_sq, im_sq, count, stripe, previous_stripe, last_added, frac, i, id)
        for (j = 0; j < N; j++)
        {

            id = k * N + j;
            re_sq = re_Z[id] * re_Z[id];
            im_sq = im_Z[id] * im_Z[id];
            i = 0.;
            count = 0.;
            stripe = 0;
            previous_stripe = 0;
            //printf("\n\nid: %d k: %d j: %d i: %d", id, k, j, i);
            while ((re_sq + im_sq < R) && (i < n_iter))
            {
                im_Z[id] = 2. * re_Z[id] * im_Z[id] + im_C[k];
                re_Z[id] = re_sq - im_sq + re_C[j];
                re_sq = re_Z[id] * re_Z[id];
                im_sq = im_Z[id] * im_Z[id];

                i++;
                //printf("\nid: %d k: %d j: %d i: %d", id, k, j, i);

                if (i > skip)
                {
                    last_added = 0.5 + 0.5 * sin(stripe_density * atan(im_Z[id] / re_Z[id]));
                    stripe += last_added;
                    count++;
                }
            }
            if (i == n_iter)
            {
                mix[id] = -1;
                iters[id] = (float)i + 1.0 + log2(log(R) / 2.0 / abs(log(re_sq + im_sq)));
            }
            else
            {
                previous_stripe = (stripe - last_added) / (count - 1.0);
                stripe = stripe / count;


                iters[id] = (float)i + 1.0 + log2(log(R) / 2.0 / abs(log(re_sq + im_sq)));
                frac = iters[id] - (int)iters[id];
                mix[id] = frac * stripe + (1.0 - frac) * previous_stripe;
            }
        }
    }
    return make_pair(iters, mix);
}


/*
PYBIND11_MODULE(fast_fractals, m) {
m.def("fast_mandel", &make_mandel, R"pbdoc(
        Compute the angles of the orbits of the selected region of the complex plane
    )pbdoc");

#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif
}
 */