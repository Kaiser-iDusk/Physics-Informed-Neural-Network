<h1><b>Physics Informed Neural Network</b></h1>

<p>The project is ongoing and is a part of the course Computational Physics. The problem statement is to solve a differential equation (1D Time-Independent Schrodinger Equation) for any given energy state which is the eigenvalue of the equation.</p>
<p>The model architecture and further details will be uploaded here shortly and the webapp is made using Django and deployed on AWS EC2 instance the link of which is given below.</p>

<h1>How to use?</h1>

<h3>$-\frac{\hbar ^ 2}{2m} \frac{d \psi}{dx} = E \psi$</h3>
<p>where E = $\frac{n^2 \hbar^2 \pi^2}{2m}$ and n = 1, 2, 3, ...</p>

<p>So the deployed version takes 2 inputs: 
<ol>
  <li>N : The number of collocation points. It basically refers to the number of sampling points over $[0, 1]$. Currently we sample points at uniform distance from each other.</li>
  <li>n : The energy state that you want to be solved. Currently the PINN is trained till n = 6. We are currently training the model to n = 10. Further training would require extensive GPU application which we currently don't have :-(</li>
</ol>

<p>That's all there is to it. Click on Predict and it would probably give you a nice graph that satifies the solution of the Schrodinger equation at a given point n.</p>

<h1>Link</h1>

<p><a href="https://tinyurl.com/pinn-demo0">Physics Informed Neural Network</p>

<h1>Foot Note</h1>

<p>Any suggestions on model improvement, optimization and other stuff is highly appreciated. Software related stuff is always welcome as well!</p>
</p>
