function [ Fig ] = FormatFigure( Fig, Width, AspectRatio, varargin )
    %function [ Fig ] = FormatFigureDiss( Fig, Width, AspectRatio )
    %   Fig: Handle of figure
    %   Width:     Widht of Paper Size in [cm] (default: 8cm)
    %   AspectRatio: Ratio of Widht/Height (default: 8/6)
    % Optional parameters:
    % MarkerSize         : Default: 12
    
    p=inputParser;
    addOptional(p, 'Fig', gcf);
    addOptional(p, 'Width', 8);
    addOptional(p, 'AspectRatio', 8/6);
    addParameter(p, 'MarkerSize', 12, @(x)isnumeric(x));
    
    if ~exist('Fig', 'var')
        Fig = gcf;
    end
    
    if ~exist('Width', 'var')
        Width = 8;
    end
    
    if ~exist('AspectRatio', 'var')
        AspectRatio = 8/6;
    end
    
    if ~isgraphics(Fig, 'figure')
        Fig = gcf;
    end
    
    parse(p, Fig, Width, AspectRatio, varargin{:});
    
    
    
    Fig.Units = 'pixels';
    
    Fig = p.Results.Fig;
    AspectRatio = p.Results.AspectRatio;
    Width = p.Results.Width;
       
    Height = Width / AspectRatio;
    ScreenSizeFactor = 100;  % in Pixels/cm
    
    ChartFontSize = 25;    % Point
    ChartLineWidth = 2;
    ChartMarkerSize = p.Results.MarkerSize;
    
    % Make this figure the current one
    figure(Fig);
    
    % Set figure position
    Pos = Fig.Position;
    Pos(3) = Width * ScreenSizeFactor;
    Pos(4) = Height * ScreenSizeFactor;
    
    Fig.PaperUnits = 'centimeters';
    Fig.PaperSize = [ Width Height];
    Fig.PaperPositionMode = 'auto';
    Fig.Position = Pos;
    
    
    
    
    % Adjust the font size on all Axes
    ax = findall(Fig,'Type','axes');
    for n=1:length(ax)
        curAx = ax(n);
        curAx.FontUnits = 'points';
        curAx.FontSize = ChartFontSize;
        curAx.LineWidth = 2;
        curAx.XRuler.Axle.LineWidth = 2;
        curAx.YRuler.Axle.LineWidth = 2;

        
        box on;
        
        leg = findobj(curAx, 'Type', 'Legend');
        
        if ~isempty(leg)
            leg.FontSize = ChartFontSize;
        end
        
        
        markers = findobj(curAx, 'Type', 'marker');
        set(markers, 'MarkerSize', ChartMarkerSize);
        
        lines = findobj(curAx, 'Type', 'line');
        set(lines, 'LineWidth', ChartLineWidth);
        set(lines, 'MarkerSize', ChartMarkerSize);
		
		% Format errorbars...
		errBars = findobj(curAx, 'Type', 'errorbar');
		set(errBars, 'LineWidth', ChartLineWidth);
		set(errBars, 'CapSize', 5*ChartLineWidth);
        
        stems = findall(curAx, 'Type', 'stem');
        set(stems, 'LineWidth', ChartLineWidth);
        
        % Check if there is a ColorBar
        cBar = findobj(curAx, 'Type', 'ColorBar');
        if length(cBar) == 1     
            cBar.FontSize = ChartFontSize;
            cL = cBar.Label;
        end
    end
    
    TiHightDelta = 0;
    TiWidthDelta = 0;
    
    % Check if there is a ColorBar
    figCbar = findobj(Fig, 'Type', 'ColorBar');
    if length(figCbar) == 1     
        figCbar.FontSize = ChartFontSize;
        cL = figCbar.Label;
    end
    
    
    for n =1:length(ax)
        
        curAx = ax(n);
        Fig.CurrentAxes = curAx;
        
        % Check if there is a ColorBar
        cBar = findobj(curAx, 'Type', 'ColorBar');
        
        % Get the size of the figure, without border and colorbar
        TI = curAx.TightInset;
        LI = curAx.LooseInset;
        
        if length(cBar) == 1
            % Make space for the colorbar label on the right
            TI(3) = TI(3) + 0.1*8/Width;
            TI(4) = TI(4) + 0.05*6 / (Width / AspectRatio);
        end
        
        % Workaround for matlab bug: If a subplot contains a colorbar, matlab seems to ignore the Loose Inset property
        % on normal graphs
        cH = findobj(curAx, 'Type', 'Contour');
        if ~length(figCbar) > 0 || length(cH) > 0
            curAx.LooseInset = TI;
        else
            disp('Ignoring LooseInset since a subplot-colorbar was detected');
        end
        %set(gca,'LooseInset',get(gca,'TightInset'));
        
        
        
        
        grid on; grid minor; 
        
    end
    
    %Pos = [5 5 Width*ScreenSizeFactor Height*ScreenSizeFactor];
    
    movegui(gcf, 'north');
    
end

