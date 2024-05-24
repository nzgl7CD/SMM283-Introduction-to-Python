<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** This layout reused from an other project Sebastian Veum has created.
*** Sebastian is the author of the original version with some inspiration from the web.
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
        <li><a href="#experimental-decoration">Experimental decoration</a></li>
        <li><a href="#tests-for-the-project">Tests for the project</a></li>
        <li><a href="#typing">Typing</a></li>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#technologies">Technologies</a></li>
        <li><a href="#functionalities">Functionalities</a></li>
        <li><a href="#design">Design</a></li>
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
        <li>
          <a href="#test-frontend-cypress-and-unit">Test frontend</a>
          <ul>
            <li><a href="#cypress-test">Cypress test</a></li>
            <li><a href="#unit-test">Unit test</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#starting">Starting</a></li>
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

#### End-2-End Testing

The team chose to use cypress to implement end-to-end testing. Cypress is a end-to-end testing framework that enables developers to create web test automation in javaScript and typescript. The framework provides various utilities for testing the DOM elements as well as the graphql requests being sent from the client side. Cypress gives the possibility to create test cases for all the functionalities implemented in react which is also why the team chose to use this framework.

To run the test in cypress’ browser and environment, simply `cd frontend` and type `npm run cypress:open` in the terminal.

The team has created test cases for asserting if redux stores are correctly updated when users search or filter the movies. The test case asserts also whether correct filtered data is given from the backend by intercepting the request and its’ response, and also comparing it to the search queries provided.

It also created tests for checking if pagination works as intended by checking if the redux store is correctly updated. The display of a detailed view of a single movie is also tested by implementing an example scenario where the first movie is selected in the test. The test case checks if correct data is being displayed after the selection.

Login and registration is tested by checking if the application correctly stores the jwt token in sessionStorage and if redux is correctly updated as well.

Lastly, reviews are tested by checking if only logged in users are allowed to publish a new review and if this new review was successfully published or not.

All test cases check if the correct DOM-elements are loaded for each component when the component is displayed. For example, when a user clicks og `Log in`, the test suite for Login and Registration checks if the correct form is loaded on the page with correct buttons. More details about the tests can be seen in the files, `fronted/cypress/integration`. The test case should be descriptive enough to explain their intent and what exactly they are testing.


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
