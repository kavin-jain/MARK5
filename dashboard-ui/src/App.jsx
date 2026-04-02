import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Hero from './pages/Hero';
import Portfolio from './pages/Portfolio';
import Analysis from './pages/Analysis';
import OrderBook from './pages/OrderBook';
import Models from './pages/Models';
import Engineering from './pages/Engineering';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Hero />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/orderbook" element={<OrderBook />} />
          <Route path="/models" element={<Models />} />
          <Route path="/engineering" element={<Engineering />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
