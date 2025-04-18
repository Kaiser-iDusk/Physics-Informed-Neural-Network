from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from PiNN.ML_model.predictions import PINN, Inference

# Create your views here.
def index(request):
    return render(request, "index.html")

def process(request):
    context = {
                "done": False,
            }

    if request.method == "POST":
        N = request.POST.get('N')
        n = request.POST.get('n')
        
        inference = Inference()
        catch = inference.infer("PiNN/ML_model/PINN.pkl", N, n)

        if catch:
            context["done"] = True
            context["image_path"] = f"staticfiles/Images/Out_{n}.png"

            return render(request, "index.html", context)

    return render(request, "index.html")
