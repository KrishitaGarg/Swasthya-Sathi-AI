import React from "react";
import { motion } from "framer-motion";
import "./Footer.css";

const Footer = () => {
  return (
    <section id="contact">
      <footer className="footer">
        <motion.div
          className="footer-content"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          <h2 className="footer-title">
            Empowering Healthcare, One Click at a Time!
          </h2>
          <p className="footer-description">
            Connect with us for innovative AI-driven medical solutions.
          </p>
          <div className="contact-info">
            <a href="mailto:krishita2005@gmail.com" className="contact-link">
              📧 contact@swasthyasathi.ai
            </a>
            <a
              href="https://swasthya-sathi-ai.streamlit.app/"
              className="contact-link"
            >
              🌐 www.swasthyasathi.ai
            </a>
          </div>
          <p className="footer-copy">
            &copy; 2025 Swasthya Sathi AI. All rights reserved.
          </p>
        </motion.div>
      </footer>
    </section>
  );
};

export default Footer;
