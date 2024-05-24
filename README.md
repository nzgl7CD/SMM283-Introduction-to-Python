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
    <a href="https://gitlab.stud.idi.ntnu.no/it2810-h21/individual/sebasv-4b"><strong>Explore the docs»</strong></a>
    <br />
    <a href="https://gitlab.stud.idi.ntnu.no/it2810-h21/individual/sebasv-4b/-/issues">Issues</a>
    ·
    <a href="https://gitlab.stud.idi.ntnu.no/it2810-h21/individual/sebasv-4b/-/commits/master">Commits</a>
    ·
    <a href="https://gitlab.stud.idi.ntnu.no/it2810-h21/individual/sebasv-4b/-/branches">Branches</a>
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

As me and my group invested extra time on perfecting the project did I face some challenges finding potential improvements for the project. Except refactoring backend from JavaScript to TypeScript. However, I was quite motivated to refactoring backend as I figuered this would be an effective approch to learn TypeScript even better. I also decided to choose a project which offered to let me focus on backend only. 

### Refactored backend

The backend itself offered a great amount of more advanced components, such as [authorisation](https://github.com/nazgul735/netflixCataloge/blob/master/backend/src/util/validateAuth.ts) validation for users to register and login, to enable all features our product offers, such as creating new reviews. The converting did therefore charge a fair amount of time. However, the final product wasn´t the main reason I spent this much of time. The first edition of the project was the main time consumer.

### Experimental decoration

At the end of the project did another member and myself begin refactoring the whole backend by using an experimental syntax of TypeScript with TC39. This sort of syntax is not fully ratified into the JavaScript specification but may offer more advanced features as it´s offered by a lower level of documentation and problem-solving forums. This also means that the implementation version in TypeScript may differ from the implementation in JavaScript when it it decided by TC39. Even though, I managed to refactor the whole backend with third party libraries as TypeGoose and make it compile did a few, foreign errors occur. Below is a screenshot of one of the schemas with experimental decoration and props from TypeGoose.

<br/>

<img src="docs/images/experimental_schema.png" alt="Searching" width="300">

<br/>

 After searching and asking for help by several experts I had to accept the fact that I couldn't complete the project and started from scratch. I had to start all over again by doing it properly in a more common manner. Even with failure did I learn a lot trying to use more advanced methods and won´t suggest it should be regretted. I may try continuing completing the main project later this year. If reading the code is of desire visit the branch, have a look: [Link](https://github.com/nazgul735/netflixCataloge/tree/experimental).


### Tests for Project 

The application is mostly tested from the project. As I mentioned earlier are several unit tests written as well as a detailed cypress test. The whole code is therefore well covered already, with above 90% jest coverage. I had to rewrite the backend util tests as they met some issues related to the new TypeScript utils they are testing. I also did a few changes to make the tests pointier. I did want to refactor the tests as well but after discussing this with a student assistant did, we conclude that it would be sort of a wate of time as I already have proven my refactoring skills.

### Typing 

The new backend is mostly typed properly as a TypeScript application should. However are a few elements typed as "any", which is someting I strived not to occur. In the resolvers, for example, does the mutation of registration of new users split in the document object of the database, id and the token. I didn't manage to type these for some reason. I spend fair amount of time to figure it out and finally gave it up for any-type. 

<br/>

<img src="docs/images/resdoc.png" alt="Searching" width="450">

<br/>

Another part I'm not happy about are the util typings. I intended to do as the screenshot below shows, which were working. However, the tests failed as $ in a type resulted with *unknown javascript syntax error* in the test files, as they are JavaScript. 

</br>
<img src="docs/images/betterTypes.png" alt="Searching" width="450">

</br>

### Built With

- [React.js](https://reactjs.org/)
- [TypeScript](https://www.typescriptlang.org/)
- [JavaScript](https://www.javascript.com/)
- [GraphQL](https://www.apollographql.com/docs/)
- [Mongoose](https://mongoosejs.com/)
- [MongoDB](https://www.mongodb.com)
- [Node.js](https://node.com)
- [Babel](https://babeljs.io/)

<div align="right">(<a href="#top">back to top</a>)</div>


### Technologies

Some technologies were required to use for the project and are therefore pre decided. With that is the interface based on react syntax. The project itself is initialized with create-react-app, npx create-react-app, implemented in TypeScript.

Regarding the GraphQL database it was clearly Apollo we wanted to use as the service provides a vast open-source library that is extremely helpful in implementing GraphQL for JavaScript applications, as the backend is written in js.

#### Redux

For state management we decided to try redux, as it is more natural to use with functional programming. However, all members personally desired to learn Redux, even though MobX is easier to wrap your head around.

#### MongoDB

As for databases, did we decide to use a nonSQL-database. Documentation naturally was decided to be stored in JSON, and therefore it became obvious to use MongoDB. As this service fits our preferences accurately, MongoDB offers several useful features such as MongoDB Compass for local launch of backend. As a pluss didn’t we need to worry about the data structure such as the number of fields or types of fields to store values.

#### Mongoose

For Node to translate between objects in code and the representation of those objects in MongoDB is an issue easily handled by mongoose. As MongoDB is schema-less Mongoose provides definitions of schema for the data models in order to specific document structure with predefined data types. Mongoose also makes returning updated documents or query results quite easier.

#### Apollo GraphQL Server/API and Client

The external library, Apollo server, was used to set up the server in the backend for the graphql APIs, meaning queries and mutations. The library, Apollo client, was used in the frontend to make API calls to the backend. Apollo client provides built in react hooks for graphql queries and mutations that are easy for use and easy to learn. The group chose to use Apollo server as this external component provides user friendly interface for testing your graphql mutations and queries. The server was instantiated with the defined [resolvers](https://www.apollographql.com/docs/apollo-server/data/resolvers/) and [schema](https://www.apollographql.com/docs/tutorial/schema/) for the queries and mutations that were made in the application.

#### JSON Web Token

In order to make all users unique and secure that only valid users may write reviews of movies did we find it necessary to relate each user to a token. JSON Web Token (jwt) gave us the opportunity to easily do so. This feature is used in authorization validations and register modules of the application.

<div align="right">(<a href="#top">back to top</a>)</div>

### Functionalities

In order to satisfy the requirements for creating an application that provides opportunity for the user to view a large, paginated dataset but also filter and search on the dataset, the group chose to solve this by implementing a movie database that displays 95 movie records. The dataset was extracted from an existing json file from an open source github repo: [link](https://github.com/FEND16/movie-json-data/blob/master/json/movies-coming-soon.json).  
The group chose to create an application on movies since it was possible to find existing data which reduced the time from set-up to development substantially.

The team chose to use redux for storing state that needed to be accessed by multiple components. In this case, the component `Appbar`, `Movies` and `FilterModal` needed access to the same states being the search queries (genre, year and search string on title). Thus, the team chose to implement the search queries as a state in redux store. This way, any components can update the search queries and use the stored search queries. The same is applied for the login functionality, where both the `Appbar`, `Login` and `Registration` component needs access to a common state that expresses whether a user is logged in or not.

#### Movies

All movies are displayed in the application by default when a user visits the home page. Any user can see the displayed movies without being authenticated.
As a user, it is possible to view movies and choose a page to load new dataset of movies. Each page displays 12 movies each. For instance, the first page displays the first 12 movies, the second page displays the 12 next and so on. Thus, the group decided to use offset limit based pagination to implement this functionality. The frontend updates the offset whenever a new page is selected. The reason behind the chosen design solution, is because it is easy to understand and implement in the client side, reduces overhead and it works on dataset that are static, meaning that will not be changed as in our case.

The team decided to implement this functionality by designing a single graphql query that takes in search queries optionally (selected genre, search string on title, and year range) which returns either filtered movies based on the provided parameters or display all movies from the database if no search queries were given as parameters. The frontend makes a call to this query for displaying the correct dataset of movies using the built in useQuery hook provided from Apollo Client.

The view of movies is implemented as a react component called `Movies`. The component ensures sending the correct updated search queries and updated offset to the backend to dynamically display the correct dataset.

Finally, it is possible as a user to see a detailed view of a single movie by clicking on “Learn more” on any movie. This button redirects users to a new page displaying more details about the chosen movie and also the reviews that have been provided to the movie. This functionality is implemented by creating a new graphql query which returns the data of a single movie based on inputted movie id. This query is called on the client side in the react component, `detailedMovie` which also displays the reviews for the movie and also title, storyline, picture and actors.

<div align="right">(<a href="#top">back to top</a>)</div>

#### Pagination, filter and search

As mentioned in the previous section, it is possible for a user to filter the movies based on genre, year or search for movies on their titles. This can be done by the search bar and the `Filter` button in the appbar. When the filter button is clicked, the app displays a model giving the user the possibility to fill in the genre or the time range the user would like to filter on or both.
The component where the filtering functionality is implemented, `FilterModal`, ensures to update the redux store with new search queries which the component `Movies` can access to return the correct filtered dataset of movies.

#### Login and registration

As a user, it is possible to create a new user and log in as a user. Authenticated users are the only role that are allowed to create reviews for each movie. The authentication of users is realized by linking a user to a jwt token which is provided to the client side if login is successful. The implementation of login in the backend is realized by the graphql query, `login` while on the client side, the login form and the request to the `login` query is implemented in the react component, `Login`. The client side displays the correct error message from the backend if login was invalid based on provided username and password. If the user is successfully logged in, the app stores the provided jwt token in sessionStorage that can be easily accessed by other components that need to read this to provide authorization control.

The review functionality is implemented as a graphql mutation in the backend that returns a valid jwt token and stores the new user in the database as a new mongodb document when valid parameters are given (username, email, password and confirmed password). The registration form and the API call to the graphql mutation for registering a new user is implemented in the react component, `Registration`. When registration is successful, the app stores the returned jwt token form the backend in sessionStorage which ensures logging in the user.

#### Review

Reviews for a given movie are accessible by anyone visiting the site. These can be viewed by visiting a single movie as mentioned previously. When a user is logged in, the user can publish a new review and the form gets visible. To publish a review, the client sends the jwt token as an authorization header which is validated in the backend. If everything works as expected, the review is successfully created and can be viewed on the detailed movie page. If invalid token is given, the app restricts the user from publishing the review.

### Design

Our overall design focuses on being simple, accessible and clean. To do this we have chosen to use Material UI for most of our components, and also adhered to the material design guidelines ([link](https://material.io/design/guidelines-overview)) which Material UI complements nicely.

Buttons, movieCard, the pagination bar, textfields and appbar are all using Material UI at the core, with customization by us to make it fit in with our red color palette and to make it responsive. When designing there was a big focus on responsiveness and to not make the site any less functional and to not make it less usable by being cluttered. We have mostly made the site responsive using media query and the built in functionality from MUI.

The data represented, movies, are made with big cards highlighting just the movie poster and its title on the front page. This makes it clean and easy to devour many elements without the site being too confusing or cluttered. When clicking on a specific movie the user is redirected to another page which highlights more details about that specific movie, as well as its corresponding reviews. This page is more detailed, but structured so that the information is segmented and easily accessible. It is also possible for the user to write their own reviews here, if they are logged in.

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

#### Frontend Testing

As well as having end to end testing we have also included component-testing using jest and the react testing library on the frontend. We have tested each individual component separately, without integrating with other components. This is to snatch up errors which only exist in one single component. The tests are mainly made to see if the content and functionality renders properly and is not crashing.
These tests do cover all the components, but not how they work together. To test the flow of the application and how the components interact with each other and the backend we have used Cypress end to end testing which will be documented below after backend testing.

#### Backend Testing

As for testing for backend modules would the most essential parts be covered by testing the util folder (.\backend\util) as it contains validators for authorization, registration and login and creating of movie queries. The testing covers the parts that throw errors when illegal arguments are set. We have also tested if legal arguments pass through and make the function return what it’s supposed to.

<div align="right">(<a href="#top">back to top</a>)</div>

<!-- RUNNING TESTS -->

## Run tests

In order to test the application, please follow given steps.

### Prerequisites

Install correct node version through your terminal.

- npm

  ```sh
  npm install npm@latest -g
  ```

- jest and cypress
  ```sh
  npm install
  ```

### Test backend

1. Navigate to backend
   ```sh
   cd backend
   ```
2. Test backend
   ```sh
   npm test
   ```

### Test frontend (Cypress and Unit)

#### Cypress test

1. Navigate to frontend
   ```sh
   cd frontend
   ```
2. Start frontend
   ```sh
   npm start
   ```
3. Open an new terminal
4. Run cypress
   ```sh
   npm run cypress:open
   ```

You may now choose which test case to run in cypress' browser

#### Unit test

1. Navigate to frontend
   ```sh
   cd frontend
   ```
2. Test frontend
   ```sh
    npm test
    ```
   <div align="right">(<a href="#top">back to top</a>)</div>

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
   git clone https://gitlab.stud.idi.ntnu.no/it2810-h21/individual/sebasv-4b
   ```
2. Install NPM packages
   ```sh
   npm install
   ```

### Starting

</br>

#### Run backend (Relevant for project 4)

1. Navigate to backend
   ```sh
   cd backend
   ```
2. Install NPM packages
   ```sh
   npm install
   ```

3. Run backend
   ```sh
   npx tsc; node dist/app.js
   ```

#### Run frontend

1. Open a new terminal

2. Navigate to frontend

```sh
 cd frontend
```

3. Run frontend

```sh
 npm start
```

<div align="right">(<a href="#top">back to top</a>)</div>

<!-- USAGE -->

## Usage

The product is a movie database. By typing in the searching bar could users search for specific movies.

</br>

<div align="left">
  <img src="docs/images/searching.png" alt="Searching" width="180" height="60">
</div>

</br>

If users aren't entirely sure which movie to look for is the filter feature a possibility. Genre, year of production and actors are all variables to choose from.

 </br>

 <div align="left">
  <img src="docs/images/filter.png" alt="Searching" width="250" height="60">
</div>

</br>

As a user should registration and login be offered. In upper right corner is the login appearing. Click on it to get started.

</br>

 <div align="left">
  <img src="docs/images/login.png" alt="Searching" width="120" height="40">
</div>

</br>

_For more info, please refere to [Documentation](https://gitlab.stud.idi.ntnu.no/it2810-h21/team-38/prosjekt-3/-/tree/master/docs)_

<br/>

<!-- CONTACT -->

## Contact

Sebastian - [@github_nazgul735](https://github.com/nazgul735) - sebasv@stud.ntnu.no

Project Link: [Project](https://github.com/nazgul735/netflixCataloge)

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
