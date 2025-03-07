import React, { useState } from "react";
import "./Navbar.css";

const Navbar = () => {
  const [showMenu, setShowMenu] = useState(false);

  const toggleMenu = () => {
    setShowMenu(!showMenu);
  };

  return (
    <div className="navbar-container">
      <nav className="navbar">
        <div className="logo-text">
          <a href="#hero" className="logo-heading">
            Swasthya Sathi AI
          </a>
        </div>
        <div className="menu-toggle" onClick={toggleMenu}>
          ☰
        </div>
        <ul className={`nav-links ${showMenu ? "show" : ""}`}>
          <li>
            <a href="#features">Why Us?</a>
          </li>
          <li>
            <a href="#working">How It Works?</a>
          </li>
          <li>
            <a href="#enhancements">Coming Up</a>
          </li>
          <li>
            <a href="#contact">Contact Us</a>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Navbar;
