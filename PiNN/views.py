from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from PiNN.ML_model.predictions import PINN, Inference

# Create your views here.
def index(request):
    return render(request, "index.html")

def about(request):
    return render(request, "about.html")

def process(request):
    mode = {'point_estimate': 1, 'range_estimate': 2}
    context = {
                "done": False,
            }

    if request.method == "POST":
        type = request.POST.get('estimation_type')
        x = request.POST.get('x')
        N = request.POST.get('N')
        n = request.POST.get('n')
        
        inference = Inference()
        catch = inference.infer("PiNN/ML_model/PINN_modified.pkl", mode[type], x, N, n)

        if catch:
            context["done"] = True
            context["image_path"] = f"media/Images/Out_{n}_{mode[type]}.png"

            return render(request, "index.html", context)

    return render(request, "index.html")