CREATE DATABASE IF NOT EXISTS School;
USE School;

CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    StudentName VARCHAR(50),
    Age INT,
    Email VARCHAR(100) UNIQUE,
    EnrollmentDate DATE,
    Major VARCHAR(50)
);

CREATE TABLE Instructors (
    InstructorID INT PRIMARY KEY,
    InstructorName VARCHAR(50),
    Email VARCHAR(100) UNIQUE,
    Department VARCHAR(50),
    HireDate DATE
);

CREATE TABLE Courses (
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(50),
    Credits INT,
    InstructorID INT,
    StartDate DATE,
    Department VARCHAR(50),
    FOREIGN KEY (InstructorID) REFERENCES Instructors(InstructorID)
);

CREATE TABLE Enrollments (
    EnrollmentID INT PRIMARY KEY,
    StudentID INT,
    CourseID INT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);

CREATE TABLE Grades (
    GradeID INT PRIMARY KEY,
    EnrollmentID INT,
    Grade VARCHAR(2),
    GradeDate DATE,
    FOREIGN KEY (EnrollmentID) REFERENCES Enrollments(EnrollmentID)
);

-- Students (52 students)
INSERT INTO Students (StudentID, StudentName, Age, Email, EnrollmentDate, Major) VALUES
(1, 'Alice', 20, 'alice@school.com', '2023-09-01', 'Computer Science'),
(2, 'Bob', 21, 'bob@school.com', '2023-09-01', 'Biology'),
(3, 'Charlie', 19, 'charlie@school.com', '2024-01-15', 'History'),
(4, 'Diana', 22, 'diana@school.com', '2022-09-01', 'Art'),
(5, 'Eve', 23, 'eve@school.com', '2023-01-10', 'Mathematics'),
(6, 'Frank', 20, 'frank@school.com', '2024-02-01', 'Physics'),
(7, 'Grace', 21, 'grace@school.com', '2023-09-15', 'Chemistry'),
(8, 'Henry', 19, 'henry@school.com', '2024-03-01', 'English'),
(9, 'Isabel', 22, 'isabel@school.com', '2023-08-20', 'Psychology'),
(10, 'Jack', 20, 'jack@school.com', '2024-01-10', 'Economics'),
(11, 'Kelly', 24, 'kelly@school.com', '2022-08-15', 'Sociology'),
(12, 'Liam', 18, 'liam@school.com', '2024-09-01', 'Undecided'),
(13, 'Mia', 19, 'mia@school.com', '2023-09-10', 'Computer Science'),
(14, 'Noah', 22, 'noah@school.com', '2022-10-01', 'Biology'),
(15, 'Olivia', 20, 'olivia@school.com', '2024-02-15', 'History'),
(16, 'Peter', 21, 'peter@school.com', '2023-08-25', 'Art'),
(17, 'Quinn', 23, 'quinn@school.com', '2023-01-20', 'Mathematics'),
(18, 'Rose', 20, 'rose@school.com', '2024-03-10', 'Physics'),
(19, 'Sam', 19, 'sam@school.com', '2023-09-05', 'Chemistry'),
(20, 'Tina', 22, 'tina@school.com', '2022-11-01', 'English'),
(21, 'Uma', 21, 'uma@school.com', '2024-01-25', 'Psychology'),
(22, 'Victor', 20, 'victor@school.com', '2023-08-15', 'Economics'),
(23, 'Wendy', 24, 'wendy@school.com', '2022-09-20', 'Sociology'),
(24, 'Xander', 18, 'xander@school.com', '2024-09-05', 'Undecided'),
(25, 'Yara', 19, 'yara@school.com', '2023-10-01', 'Computer Science'),
(26, 'Zane', 22, 'zane@school.com', '2022-12-01', 'Biology'),
(27, 'Ava', 20, 'ava@school.com', '2024-02-20', 'History'),
(28, 'Ben', 21, 'ben@school.com', '2023-09-25', 'Art'),
(29, 'Clara', 23, 'clara@school.com', '2023-02-01', 'Mathematics'),
(30, 'Dan', 20, 'dan@school.com', '2024-03-15', 'Physics'),
(31, 'Ella', 19, 'ella@school.com', '2023-10-10', 'Chemistry'),
(32, 'Finn', 22, 'finn@school.com', '2022-11-15', 'English'),
(33, 'Gina', 21, 'gina@school.com', '2024-02-01', 'Psychology'),
(34, 'Hank', 20, 'hank@school.com', '2023-08-30', 'Economics'),
(35, 'Ivy', 24, 'ivy@school.com', '2022-10-05', 'Sociology'),
(36, 'Jonah', 18, 'jonah@school.com', '2024-09-10', 'Undecided'),
(37, 'Kara', 19, 'kara@school.com', '2023-11-01', 'Computer Science'),
(38, 'Leo', 22, 'leo@school.com', '2022-12-15', 'Biology'),
(39, 'Mila', 20, 'mila@school.com', '2024-03-01', 'History'),
(40, 'Nate', 21, 'nate@school.com', '2023-09-30', 'Art'),
(41, 'Omar', 23, 'omar@school.com', '2023-02-15', 'Mathematics'),
(42, 'Pia', 20, 'pia@school.com', '2024-03-20', 'Physics'),
(43, 'Raul', 19, 'raul@school.com', '2023-11-10', 'Chemistry'),
(44, 'Sara', 22, 'sara@school.com', '2022-12-20', 'English'),
(45, 'Tom', 21, 'tom@school.com', '2024-02-10', 'Psychology'),
(46, 'Uma', 20, 'uma2@school.com', '2023-09-15', 'Economics'),
(47, 'Vera', 24, 'vera@school.com', '2022-10-15', 'Sociology'),
(48, 'Will', 18, 'will@school.com', '2024-09-15', 'Undecided'),
(49, 'Xena', 19, 'xena@school.com', '2023-11-20', 'Computer Science'),
(50, 'Yuri', 22, 'yuri@school.com', '2022-12-25', 'Biology'),
(51, 'Zoe', 20, 'zoe@school.com', '2024-03-05', 'History'),
(52, 'Adam', 21, 'adam@school.com', '2023-10-05', 'Art');

-- Instructors (10 instructors)
INSERT INTO Instructors (InstructorID, InstructorName, Email, Department, HireDate) VALUES
(1, 'Dr. Smith', 'smith@school.com', 'Mathematics', '2018-06-01'),
(2, 'Prof. Jones', 'jones@school.com', 'Science', '2019-08-15'),
(3, 'Dr. Lee', 'lee@school.com', 'History', '2020-01-10'),
(4, 'Ms. Taylor', 'taylor@school.com', 'Fine Arts', '2021-03-20'),
(5, 'Dr. Brown', 'brown@school.com', 'Physics', '2017-09-01'),
(6, 'Prof. Adams', 'adams@school.com', 'Science', '2018-11-05'),
(7, 'Ms. Carter', 'carter@school.com', 'English', '2022-02-15'),
(8, 'Mr. Wilson', 'wilson@school.com', 'Computer Science', '2020-07-01'),
(9, 'Dr. Evans', 'evans@school.com', 'Psychology', '2019-04-10'),
(10, 'Prof. Patel', 'patel@school.com', 'Economics', '2021-09-01');

-- Courses (12 courses)
INSERT INTO Courses (CourseID, CourseName, Credits, InstructorID, StartDate, Department) VALUES
(101, 'Math', 3, 1, '2025-01-10', 'Mathematics'),
(102, 'Science', 4, 2, '2025-01-10', 'Science'),
(103, 'History', 3, 3, '2025-01-15', 'History'),
(104, 'Art', 2, 4, '2025-02-01', 'Fine Arts'),
(105, 'Physics', 4, 5, '2025-01-20', 'Physics'),
(106, 'Chemistry', 3, 6, '2025-01-20', 'Science'),
(107, 'English Lit', 3, 7, '2025-02-10', 'English'),
(108, 'Programming', 4, 8, '2025-02-15', 'Computer Science'),
(109, 'Psychology', 3, 9, '2025-01-25', 'Psychology'),
(110, 'Economics', 3, 10, '2025-02-05', 'Economics'),
(111, 'Sociology', 3, NULL, '2025-03-01', 'Sociology'),
(112, 'Statistics', 4, NULL, '2025-03-10', 'Mathematics');

-- Enrollments (60 enrollments)
INSERT INTO Enrollments (EnrollmentID, StudentID, CourseID) VALUES
(1, 1, 101), (2, 1, 108), (3, 2, 102), (4, 2, 105), (5, 3, 103),
(6, 4, 104), (7, 5, 101), (8, 5, 106), (9, 6, 105), (10, 6, 102),
(11, 7, 106), (12, 7, 102), (13, 8, 107), (14, 9, 109), (15, 9, 107),
(16, 10, 110), (17, 10, 101), (18, 11, 111), (19, 11, 109), (20, 3, 107),
(21, 13, 108), (22, 14, 102), (23, 15, 103), (24, 16, 104), (25, 17, 101),
(26, 18, 105), (27, 19, 106), (28, 20, 107), (29, 21, 109), (30, 22, 110),
(31, 23, 111), (32, 25, 108), (33, 26, 102), (34, 27, 103), (35, 28, 104),
(36, 29, 101), (37, 30, 105), (38, 31, 106), (39, 32, 107), (40, 33, 109),
(41, 34, 110), (42, 35, 111), (43, 37, 108), (44, 38, 102), (45, 39, 103),
(46, 40, 104), (47, 41, 101), (48, 42, 105), (49, 43, 106), (50, 44, 107),
(51, 45, 109), (52, 46, 110), (53, 47, 111), (54, 49, 108), (55, 50, 102),
(56, 51, 103), (57, 52, 104), (58, 13, 101), (59, 25, 101), (60, 37, 101);

-- Grades (60 grades)
INSERT INTO Grades (GradeID, EnrollmentID, Grade, GradeDate) VALUES
(1, 1, 'A', '2025-03-15'), (2, 2, 'A+', '2025-04-01'), (3, 3, 'B+', '2025-03-20'),
(4, 4, 'B', '2025-03-25'), (5, 5, 'A-', '2025-03-18'), (6, 6, NULL, NULL),
(7, 7, 'B', '2025-03-15'), (8, 8, 'C+', '2025-03-22'), (9, 9, 'A', '2025-03-25'),
(10, 10, 'B+', '2025-03-20'), (11, 11, 'B-', '2025-03-22'), (12, 12, NULL, NULL),
(13, 13, NULL, NULL), (14, 14, 'A', '2025-03-28'), (15, 15, 'B+', '2025-04-05'),
(16, 16, 'B', '2025-04-10'), (17, 17, 'C', '2025-03-15'), (18, 18, 'A-', '2025-04-15'),
(19, 19, 'B', '2025-03-28'), (20, 20, NULL, NULL), (21, 21, 'A', '2025-04-01'),
(22, 22, 'B+', '2025-03-20'), (23, 23, 'A-', '2025-03-18'), (24, 24, NULL, NULL),
(25, 25, 'B', '2025-03-15'), (26, 26, 'A', '2025-03-25'), (27, 27, 'C+', '2025-03-22'),
(28, 28, NULL, NULL), (29, 29, 'A', '2025-03-28'), (30, 30, 'B', '2025-04-10'),
(31, 31, 'A-', '2025-04-15'), (32, 32, 'A+', '2025-04-01'), (33, 33, 'B+', '2025-03-20'),
(34, 34, 'A-', '2025-03-18'), (35, 35, NULL, NULL), (36, 36, 'B', '2025-03-15'),
(37, 37, 'A', '2025-03-25'), (38, 38, 'C+', '2025-03-22'), (39, 39, NULL, NULL),
(40, 40, 'A', '2025-03-28'), (41, 41, 'B', '2025-04-10'), (42, 42, 'A-', '2025-04-15'),
(43, 43, 'A+', '2025-04-01'), (44, 44, 'B+', '2025-03-20'), (45, 45, 'A-', '2025-03-18'),
(46, 46, NULL, NULL), (47, 47, 'B', '2025-03-15'), (48, 48, 'A', '2025-03-25'),
(49, 49, 'C+', '2025-03-22'), (50, 50, NULL, NULL), (51, 51, 'A', '2025-03-28'),
(52, 52, 'B', '2025-04-10'), (53, 53, 'A-', '2025-04-15'), (54, 54, 'A+', '2025-04-01'),
(55, 55, 'B+', '2025-03-20'), (56, 56, 'A-', '2025-03-18'), (57, 57, NULL, NULL),
(58, 58, 'B', '2025-03-15'), (59, 59, 'A', '2025-03-25'), (60, 60, 'C+', '2025-03-22');