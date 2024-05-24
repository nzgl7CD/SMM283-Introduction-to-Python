<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** This layout reused from an other project Sebastian Veum has created.
*** Sebastian is the author of the original version.
-->

<!-- PROJECT LOGO -->

<div align="center">

<h1>SMM283 Coursework</h1>

  <p>
    by Sebastian Veum
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#refactored-backend">Refactored backend</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li>
      <a href="#functionalities">Functionalities</a>
    </li>
    <li>
      <a href="#run-tests">Run tests</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#test-backend">Test backend</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

</br>

## About the Project

The project is developed as a response to the requirments of the coursework given by the professor.  

### Refactored backend

The backend itself offered a great amount of more advanced components.



### Tests for Project 

The application is mostly tested from the project. As I mentioned earlier are several unit tests written as well as a detailed cypress test. The whole code is therefore well covered already, with above 90% jest coverage. I had to rewrite the backend util tests as they met some issues related to the new TypeScript utils they are testing. I also did a few changes to make the tests pointier. I did want to refactor the tests as well but after discussing this with a student assistant did, we conclude that it would be sort of a wate of time as I already have proven my refactoring skills.
 

<div align="right">(<a href="#top">back to top</a>)</div>


### Technologies


<div align="right">(<a href="#top">back to top</a>)</div>

### Functionalities




<div align="right">(<a href="#top">back to top</a>)</div>



#### Review

Reviews for a given movie are accessible by anyone visiting the site. These can be viewed by visiting a single movie as mentioned previously. When a user is logged in, the user can publish a new review and the form gets visible. To publish a review, the client sends the jwt token as an authorization header which is validated in the backend. If everything works as expected, the review is successfully created and can be viewed on the detailed movie page. If invalid token is given, the app restricts the user from publishing the review.


<div align="right">(<a href="#top">back to top</a>)</div>

### Testing


#### Backend Testing

As for testing for backend modules would the most essential parts be covered by testing the util folder (.\backend\util) as it contains validators for authorization, registration and login and creating of movie queries. The testing covers the parts that throw errors when illegal arguments are set. We have also tested if legal arguments pass through and make the function return what it’s supposed to.

<div align="right">(<a href="#top">back to top</a>)</div>

<!-- RUNNING TESTS -->


### Test backend


<!-- GETTING STARTED -->

## Getting Started

In order to set up the project locally, please follow given steps.

### Prerequisites

Install correct node version through your terminal.

- npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/nzgl7CD/SMM283-Introduction-to-Python.git
   ```
2. Install packages
   ```sh
   pip install -r requirements.txt
   ```

#### Run code

1. Open a new terminal

2. Navigate to main folder

```sh
    python main.py
```



<div align="right">(<a href="#top">back to top</a>)</div>

<!-- USAGE -->

## Usage


## Contact

Project Link: [Project](https://github.com/nzgl7CD/SMM283-Introduction-to-Python)

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Apollo documentation](https://www.apollographql.com/docs/)
- [MongoDB documentation](https://docs.mongodb.com/)
- [hidjou Github](https://github.com/hidjou/classsed-graphql-mern-apollo/tree/master)
- [Ben Awad](https://www.youtube.com/watch?v=YFkJGEefgU8)
- [Cypress Documentation](https://docs.cypress.io/guides/overview/why-cypress)
- [MUI](https://mui.com/getting-started/usage/)
- [Typegoose Github](https://github.com/typegoose/typegoose)
- [TypeGraphQL](https://typegraphql.com/)
- [Decorator](https://www.typescriptlang.org/docs/handbook/decorators.html)


<br/>

<br/>

<br/>

<br/>

<br/>

<br/>

<br/>

<div align="center">
  <h3>Sebastian Veum</h3>
</div>

<div align="right">(<a href="#top">back to top</a>)</div>
