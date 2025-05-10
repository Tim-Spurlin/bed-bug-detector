import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Label
} from 'recharts';

const generateTimelineData = () => {
  const data = [];
  const months = 24; // 2 years
  const diyInitialCost = 100;
  const professionalCostPerVisit = 250;
  const k9CostPerVisit = 1500;
  const quarterlyVisits = 4; // visits per year
  
  let diyTotalCost = 0;
  let profTotalCost = 0;
  let k9TotalCost = 0;
  
  // Generate monthly data points
  for (let i = 0; i <= months; i++) {
    // DIY cost is one-time only at month 0
    if (i === 0) {
      diyTotalCost = diyInitialCost;
    }
    
    // Professional and K9 costs occur quarterly
    if (i > 0 && i % 3 === 0) { // Every quarter (3 months)
      profTotalCost += professionalCostPerVisit;
      k9TotalCost += k9CostPerVisit;
    }
    
    // Calculate savings
    const savingsVsProf = profTotalCost - diyTotalCost;
    const savingsVsK9 = k9TotalCost - diyTotalCost;
    
    // Calculate ROI
    const roiVsProf = diyTotalCost > 0 ? (savingsVsProf / diyTotalCost) * 100 : 0;
    const roiVsK9 = diyTotalCost > 0 ? (savingsVsK9 / diyTotalCost) * 100 : 0;
    
    data.push({
      month: i,
      DIYCost: diyTotalCost,
      ProfessionalCost: profTotalCost,
      K9Cost: k9TotalCost,
      SavingsVsProfessional: savingsVsProf,
      SavingsVsK9: savingsVsK9,
      ROIVsProfessional: roiVsProf,
      ROIVsK9: roiVsK9
    });
  }
  
  return data;
};

const ROITimelineChart = () => {
  const data = generateTimelineData();
  
  // Find breakeven points
  const breakEvenProfIndex = data.findIndex(d => d.SavingsVsProfessional > 0);
  const breakEvenK9Index = data.findIndex(d => d.SavingsVsK9 > 0);
  
  const breakEvenProfMonth = breakEvenProfIndex >= 0 ? data[breakEvenProfIndex].month : null;
  const breakEvenK9Month = breakEvenK9Index >= 0 ? data[breakEvenK9Index].month : null;
  
  const formatYAxis = (value) => {
    return `$${value}`;
  };
  
  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-center mb-4">Cumulative Cost Comparison & ROI Timeline</h2>
      
      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-2">Cost Over Time (2 Years)</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 50,
              bottom: 10,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="month" 
              label={{ value: 'Months', position: 'insideBottomRight', offset: -5 }}
              tickFormatter={(value) => value % 3 === 0 ? value : ''}
            />
            <YAxis 
              tickFormatter={formatYAxis}
              label={{ value: 'Cumulative Cost ($)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value) => [`$${value}`, '']}
              labelFormatter={(value) => `Month ${value}`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="DIYCost" 
              name="DIY Detector" 
              stroke="#B71C1C" 
              strokeWidth={3}
              activeDot={{ r: 8 }}
            />
            <Line 
              type="monotone" 
              dataKey="ProfessionalCost" 
              name="Professional Inspection" 
              stroke="#2E7D32" 
              strokeWidth={3}
              activeDot={{ r: 8 }}
            />
            <Line 
              type="monotone" 
              dataKey="K9Cost" 
              name="K9 Detection Service" 
              stroke="#1565C0" 
              strokeWidth={3}
              activeDot={{ r: 8 }}
            />
            
            {/* Breakeven reference lines */}
            {breakEvenProfMonth !== null && (
              <ReferenceLine x={breakEvenProfMonth} stroke="#2E7D32" strokeDasharray="3 3">
                <Label value="Breakeven vs Professional" position="top" fill="#2E7D32" />
              </ReferenceLine>
            )}
            
            {breakEvenK9Month !== null && (
              <ReferenceLine x={breakEvenK9Month} stroke="#1565C0" strokeDasharray="3 3">
                <Label value="Breakeven vs K9" position="insideTopLeft" fill="#1565C0" />
              </ReferenceLine>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-2">Return on Investment</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 50,
              bottom: 10,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="month" 
              label={{ value: 'Months', position: 'insideBottomRight', offset: -5 }}
              tickFormatter={(value) => value % 3 === 0 ? value : ''}
            />
            <YAxis
              label={{ value: 'ROI (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 'dataMax']}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip 
              formatter={(value) => [`${value.toFixed(0)}%`, '']}
              labelFormatter={(value) => `Month ${value}`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="ROIVsProfessional" 
              name="ROI vs Professional" 
              stroke="#2E7D32" 
              strokeWidth={3}
              activeDot={{ r: 8 }}
            />
            <Line 
              type="monotone" 
              dataKey="ROIVsK9" 
              name="ROI vs K9" 
              stroke="#1565C0" 
              strokeWidth={3}
              activeDot={{ r: 8 }}
            />
            
            {/* Breakeven reference line at 100% ROI */}
            <ReferenceLine y={100} stroke="#B71C1C" strokeDasharray="3 3">
              <Label value="100% ROI (Breakeven)" position="insideRight" fill="#B71C1C" />
            </ReferenceLine>
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="text-sm text-gray-600 mt-4">
        <p className="font-bold">ROI Analysis:</p>
        <p>
          The DIY detector breaks even against professional inspection services after just {breakEvenProfMonth} month(s) 
          and against K9 detection services after {breakEvenK9Month} month(s).
        </p>
        <p className="mt-2">
          After 2 years, the DIY detector provides an ROI of {data[data.length - 1].ROIVsProfessional.toFixed(0)}% 
          compared to professional inspections and {data[data.length - 1].ROIVsK9.toFixed(0)}% compared to K9 services.
        </p>
        <p className="mt-2">
          Total 2-year savings: ${data[data.length - 1].SavingsVsProfessional.toFixed(0)} vs professional inspections 
          and ${data[data.length - 1].SavingsVsK9.toFixed(0)} vs K9 services.
        </p>
      </div>
    </div>
  );
};

export default ROITimelineChart;
