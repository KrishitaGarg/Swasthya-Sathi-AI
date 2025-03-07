import React from "react";
import "./LandingPage.css";
import Navbar from "./Navbar/Navbar";
import Hero from "./Hero/Hero";
import Features from "./Features/Features";
import HowItWorks from "./WorkingHow/Working";
import FutureEnhancements from "./FutureEnhancements/FutureEnhancements";
import Footer from "./Footer/Footer";

const App = () => {
  return (
    <div className="landing-page">
      {/* Navbar */}
      <Navbar />

      {/* Hero Section */}
      <Hero />

      {/* Features Section */}
      <Features />

      {/* How it Works Section */}
      <HowItWorks />

      {/* Future Enhancements */}
      <FutureEnhancements />

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;
